"""
Unified Cohere inference — supports:
  - Aya Expanse       (CohereForCausalLM  / cohere)
  - Tiny-Aya-Global   (Cohere2ForCausalLM / cohere2)
  - North-Mini-Code   (Cohere2MoE         / cohere2_moe)

Detects model type automatically from config.json.

Architecture differences:
  cohere:     full attention, split-half RoPE, dense MLP
  cohere2:    hybrid sliding+full, GPT-J RoPE, dense MLP
  cohere2_moe: hybrid sliding+full, GPT-J RoPE on sliding only,
               NO RoPE on full-attention layers, MoE MLP (128 experts/8 active),
               layer 0 is dense, rest are MoE, sigmoid router

Install:
    pip install torch safetensors tokenizers pydantic
"""

import json
import re
import argparse
import getpass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from pydantic import BaseModel, field_validator, Field


# ── Pydantic Response Schema ──────────────────────────────────────────────────

class AssistantResponse(BaseModel):
    content:    str  = Field(..., description="The assistant's response text")
    truncated:  bool = Field(False)
    char_count: int  = Field(0)
    max_chars:  int  = Field(100)

    @field_validator("content")
    @classmethod
    def strip_content(cls, v: str) -> str:
        return v.strip()

    def model_post_init(self, __context):
        if len(self.content) > self.max_chars:
            cut = self.content[:self.max_chars]
            b   = cut.rfind(" ")
            self.content   = cut[:b] if b > 0 else cut
            self.truncated = True
        self.char_count = len(self.content)


# ── RoPE ──────────────────────────────────────────────────────────────────────

def build_rope_split(head_dim, base, device, dtype, offset):
    half     = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t        = torch.tensor([offset], device=device).float()
    emb      = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def build_rope_gptj(head_dim, base, device, dtype, offset):
    half     = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t        = torch.tensor([offset], device=device).float()
    emb      = torch.outer(t, inv_freq).repeat_interleave(2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope_split(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), \
           (k * cos) + (rotate_half(k) * sin)


def apply_rope_gptj(q, k, cos, sin):
    return (q * cos) + (rotate_every_two(q) * sin), \
           (k * cos) + (rotate_every_two(k) * sin)


# ── LayerNorm ─────────────────────────────────────────────────────────────────

class LayerNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps    = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x):
        mean = x.float().mean(-1, keepdim=True)
        var  = (x.float() - mean).pow(2).mean(-1, keepdim=True)
        return ((x.float() - mean) / (var + self.eps).sqrt()).to(x.dtype) * self.weight


# ── Attention ─────────────────────────────────────────────────────────────────

class Attention(torch.nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.n_heads    = cfg["num_attention_heads"]
        self.n_kv_heads = cfg.get("num_key_value_heads", self.n_heads)
        self.head_dim   = cfg.get("head_dim", cfg["hidden_size"] // self.n_heads)
        self.scale      = self.head_dim ** -0.5
        hidden          = cfg["hidden_size"]

        self.q_proj = torch.nn.Linear(hidden, self.n_heads    * self.head_dim, bias=False)
        self.k_proj = torch.nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(hidden, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.n_heads * self.head_dim, hidden,    bias=False)

        layer_types      = cfg.get("layer_types", [])
        self.is_sliding  = (layer_idx < len(layer_types) and
                            layer_types[layer_idx] == "sliding_attention")
        self.window      = cfg.get("sliding_window") if self.is_sliding else None
        self.rope_style  = cfg.get("position_embedding_type", "rope")
        self.model_type  = cfg.get("model_type", "cohere")

    def forward(self, x, cos, sin, kv_cache):
        B = x.shape[0]
        q = self.q_proj(x).view(B, 1, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, 1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # MoE model: only sliding-window layers use RoPE; full-attention layers have no pos emb
        use_rope = (self.model_type != "cohere2_moe") or self.is_sliding
        if use_rope and cos is not None:
            if self.rope_style == "rope_gptj":
                q, k = apply_rope_gptj(q, k, cos, sin)
            else:
                q, k = apply_rope_split(q, k, cos, sin)

        if kv_cache[0] is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        kv_cache[0], kv_cache[1] = k, v

        ka, va = k, v
        if self.window and k.shape[2] > self.window:
            ka = k[:, :, -self.window:, :]
            va = v[:, :, -self.window:, :]

        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            ka  = ka.repeat_interleave(rep, dim=1)
            va  = va.repeat_interleave(rep, dim=1)

        attn = F.softmax(
            (torch.matmul(q, ka.transpose(-2, -1)) * self.scale).float(), dim=-1
        ).to(q.dtype)
        out = torch.matmul(attn, va).transpose(1, 2).contiguous().view(B, 1, -1)
        return self.o_proj(out)


# ── MLP variants ──────────────────────────────────────────────────────────────

class DenseMLP(torch.nn.Module):
    """Standard dense SwiGLU MLP (used by cohere/cohere2 and layer 0 of cohere2_moe)."""
    def __init__(self, hidden, inter):
        super().__init__()
        self.gate = torch.nn.Linear(hidden, inter, bias=False)
        self.up   = torch.nn.Linear(hidden, inter, bias=False)
        self.down = torch.nn.Linear(inter, hidden, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoEMLP(torch.nn.Module):
    """
    Sparse Mixture-of-Experts MLP for cohere2_moe.
    - num_experts total, top_k activated per token
    - Router uses sigmoid (not softmax) before top-k
    - Each expert is an independent SwiGLU FFN

    Expert weights are allocated with torch.empty (no random fill) and
    requires_grad=False since we only do inference. This halves the peak
    memory during construction vs random-init + grad tensors.
    """
    def __init__(self, hidden, expert_inter, num_experts, top_k):
        super().__init__()
        self.num_experts  = num_experts
        self.top_k        = top_k
        self.hidden       = hidden
        self.expert_inter = expert_inter

        self.router = torch.nn.Linear(hidden, num_experts, bias=False)

        # Use empty (no random fill) + no grad — inference only
        # Shape: (num_experts, inter, hidden) for gate/up; (num_experts, hidden, inter) for down
        with torch.no_grad():
            self.expert_gate = torch.nn.Parameter(
                torch.empty(num_experts, expert_inter, hidden), requires_grad=False)
            self.expert_up   = torch.nn.Parameter(
                torch.empty(num_experts, expert_inter, hidden), requires_grad=False)
            self.expert_down = torch.nn.Parameter(
                torch.empty(num_experts, hidden, expert_inter), requires_grad=False)

    def forward(self, x):
        # x: (B, 1, hidden) — single token, B=1 in our use case
        B, T, H = x.shape
        x_flat = x.view(B * T, H)  # (N, hidden), N=1

        # Router: sigmoid scores, pick top_k
        scores = torch.sigmoid(self.router(x_flat))          # (N, num_experts)
        topk_scores, topk_idx = torch.topk(scores, self.top_k, dim=-1)  # (N, top_k)
        topk_scores = topk_scores / (topk_scores.sum(dim=-1, keepdim=True) + 1e-9)

        # --- Batched expert computation (no Python loops over experts) ---
        # Gather selected expert weights in one shot
        # topk_idx: (N, top_k) -> flatten to (N*top_k,)
        N       = B * T
        flat_idx = topk_idx.view(-1)                         # (N*top_k,)

        # Gather: (N*top_k, inter, hidden)
        sel_gate = self.expert_gate[flat_idx]
        sel_up   = self.expert_up[flat_idx]
        sel_down = self.expert_down[flat_idx]

        # Expand input to match: x_flat (N, hidden) -> (N*top_k, hidden)
        x_exp = x_flat.unsqueeze(1).expand(N, self.top_k, H).reshape(N * self.top_k, H)

        # SwiGLU for all selected experts simultaneously
        # gate/up: (N*top_k, inter, hidden) @ (N*top_k, hidden, 1) = (N*top_k, inter, 1)
        gate_out = torch.bmm(sel_gate, x_exp.unsqueeze(-1)).squeeze(-1)   # (N*top_k, inter)
        up_out   = torch.bmm(sel_up,   x_exp.unsqueeze(-1)).squeeze(-1)   # (N*top_k, inter)
        h        = F.silu(gate_out) * up_out                               # (N*top_k, inter)

        # down: (N*top_k, hidden, inter) @ (N*top_k, inter, 1) = (N*top_k, hidden)
        expert_out = torch.bmm(sel_down, h.unsqueeze(-1)).squeeze(-1)     # (N*top_k, hidden)

        # Weighted sum: reshape to (N, top_k, hidden), multiply by scores, sum over top_k
        expert_out  = expert_out.view(N, self.top_k, H)
        weights     = topk_scores.unsqueeze(-1)                            # (N, top_k, 1)
        out         = (expert_out * weights).sum(dim=1)                    # (N, hidden)

        return out.view(B, T, H)


class DecoderLayer(torch.nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn = Attention(cfg, layer_idx)
        self.norm = LayerNorm(cfg["hidden_size"], cfg.get("layer_norm_eps", 1e-5))

        model_type = cfg.get("model_type", "cohere")
        hidden     = cfg["hidden_size"]

        if model_type == "cohere2_moe" and layer_idx > 0:
            # MoE layers — layer 0 is always dense
            num_experts  = cfg.get("num_experts",              128)
            top_k        = cfg.get("num_experts_per_tok",        8)
            # Use sniffed expert_intermediate_size if available
            expert_inter = cfg.get("expert_intermediate_size",
                                   cfg.get("moe_intermediate_size", 768))
            self.mlp = MoEMLP(hidden, expert_inter, num_experts, top_k)
        elif model_type == "cohere2_moe" and layer_idx == 0:
            # Layer 0 is always dense — use sniffed dense_intermediate_size
            inter    = cfg.get("dense_intermediate_size",
                               cfg.get("intermediate_size", hidden * 4))
            self.mlp = DenseMLP(hidden, inter)
        else:
            inter    = cfg.get("intermediate_size", hidden * 4)
            self.mlp = DenseMLP(hidden, inter)

    def forward(self, x, cos, sin, kv_cache):
        n = self.norm(x)
        return x + self.attn(n, cos, sin, kv_cache) + self.mlp(n)


# ── Main model ────────────────────────────────────────────────────────────────

class CohereModel(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg         = cfg
        self.logit_scale = cfg.get("logit_scale", 1.0)
        self.embed       = torch.nn.Embedding(cfg["vocab_size"], cfg["hidden_size"])
        self.layers      = torch.nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        )
        self.norm        = LayerNorm(cfg["hidden_size"], cfg.get("layer_norm_eps", 1e-5))
        self.rope_base   = cfg.get("rope_theta", 10000.0)
        self.rope_style  = cfg.get("position_embedding_type", "rope")
        self.head_dim    = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        self.model_type  = cfg.get("model_type", "cohere")

    def forward_one(self, token_id, offset, kv_caches, device, dtype):
        x = self.embed(torch.tensor([[token_id]], dtype=torch.long, device=device))

        # Build RoPE — MoE full-attention layers skip it, handled inside Attention
        if self.rope_style == "rope_gptj":
            cos, sin = build_rope_gptj(self.head_dim, self.rope_base, device, dtype, offset)
        else:
            cos, sin = build_rope_split(self.head_dim, self.rope_base, device, dtype, offset)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, kv_caches[i])

        x = self.norm(x)
        return (F.linear(x, self.embed.weight) * self.logit_scale)[:, 0, :]


# ── Weight Loading ────────────────────────────────────────────────────────────

import re as _re

_EXPERT_PAT = _re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight"
)


def layer_idx_from_key(k):
    m = _re.match(r"layers\.(\d+)\.", k)
    return int(m.group(1)) if m else -1


def _remap_key(k, model_type):
    """Remap a single HF checkpoint key to our module key. Returns None for expert keys."""
    if model_type == "cohere2_moe" and _EXPERT_PAT.match(k):
        return None  # handled separately

    k = k.replace("model.embed_tokens.weight", "embed.weight")
    k = k.replace("model.norm.weight",          "norm.weight")
    k = k.replace("model.layers.",              "layers.")
    k = k.replace(".self_attn.q_proj",          ".attn.q_proj")
    k = k.replace(".self_attn.k_proj",          ".attn.k_proj")
    k = k.replace(".self_attn.v_proj",          ".attn.v_proj")
    k = k.replace(".self_attn.o_proj",          ".attn.o_proj")
    k = k.replace(".input_layernorm",           ".norm")

    if model_type == "cohere2_moe":
        k = k.replace("layers.0.mlp.gate_proj", "layers.0.mlp.gate")
        k = k.replace("layers.0.mlp.up_proj",   "layers.0.mlp.up")
        k = k.replace("layers.0.mlp.down_proj", "layers.0.mlp.down")
        if k.endswith(".mlp.gate.weight") and layer_idx_from_key(k) > 0:
            k = k.replace(".mlp.gate.weight", ".mlp.router.weight")
    else:
        k = k.replace(".mlp.gate_proj", ".mlp.gate")
        k = k.replace(".mlp.up_proj",   ".mlp.up")
        k = k.replace(".mlp.down_proj", ".mlp.down")

    return k


def load_weights(model, model_dir, cfg):
    """
    Stream-load weights shard by shard, copying directly into model parameters.
    Never holds more than one shard + per-layer expert accumulator in memory.
    Expert tensors are stacked and copied to model as soon as all 128 are seen.
    """
    model_dir  = Path(model_dir)
    shards     = sorted(model_dir.glob("*.safetensors"))
    model_type = cfg.get("model_type", "cohere")
    num_experts = cfg.get("num_experts", 128)

    if not shards:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")

    # Build a name->parameter map for O(1) lookup
    param_map = {name: param for name, param in model.named_parameters()}

    # Per-layer expert accumulator: expert_acc[layer_idx][proj] = {eid: tensor}
    # Only kept until a layer's experts are complete, then flushed to model
    expert_acc = {}

    def flush_layer_experts(li):
        """Stack and copy experts for layer li into model, then free buffer."""
        buf = expert_acc.get(li)
        if buf is None:
            return
        for proj in ["gate", "up", "down"]:
            pkey = f"layers.{li}.mlp.expert_{proj}"
            if pkey in param_map and len(buf[proj]) == num_experts:
                stacked = torch.stack(
                    [buf[proj][ei] for ei in range(num_experts)], dim=0
                )
                with torch.no_grad():
                    param_map[pkey].copy_(stacked)
                del stacked
        del expert_acc[li]

    print(f"Loading {len(shards)} shard(s)...")
    missing_keys = set(param_map.keys())

    for shard_idx, shard in enumerate(shards, 1):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for k in keys:
                # Expert tensor
                if model_type == "cohere2_moe":
                    m = _EXPERT_PAT.match(k)
                    if m:
                        li, ei, proj = int(m.group(1)), int(m.group(2)), m.group(3)
                        if li not in expert_acc:
                            expert_acc[li] = {"gate": {}, "up": {}, "down": {}}
                        expert_acc[li][proj][ei] = f.get_tensor(k)
                        # Flush as soon as all experts for this layer+proj are loaded
                        if (len(expert_acc[li]["gate"]) == num_experts and
                            len(expert_acc[li]["up"])   == num_experts and
                            len(expert_acc[li]["down"]) == num_experts):
                            flush_layer_experts(li)
                        continue

                # Regular tensor — remap and copy directly into model param
                mk = _remap_key(k, model_type)
                if mk and mk in param_map:
                    with torch.no_grad():
                        param_map[mk].copy_(f.get_tensor(k))
                    missing_keys.discard(mk)

        if shard_idx % 10 == 0 or shard_idx == len(shards):
            print(f"  {shard_idx}/{len(shards)} shards  "
                  f"({len(expert_acc)} expert layers pending)", flush=True)

    # Flush any remaining partially-loaded expert layers
    for li in list(expert_acc.keys()):
        flush_layer_experts(li)

    # Report
    # Expert param keys won't appear in missing_keys since we copy directly
    non_expert_missing = [k for k in missing_keys
                          if "expert_" not in k and "router" not in k]
    if non_expert_missing:
        print(f"  WARNING missing ({len(non_expert_missing)}): {non_expert_missing[:5]}")

    print("Weights loaded.\n")


# ── Quantization ──────────────────────────────────────────────────────────────

def quantize_model(model):
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = module.weight.data.to(torch.bfloat16)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.bfloat16)
    return model


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def load_tokenizer(model_dir):
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    tok.post_processor = None
    return tok


def get_special_ids(model_dir, cfg):
    name_to_id = {}
    cfg_path = Path(model_dir) / "tokenizer_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            tcfg = json.load(f)
        for id_str, info in tcfg.get("added_tokens_decoder", {}).items():
            name_to_id[info["content"]] = int(id_str)
    if not name_to_id:
        with open(Path(model_dir) / "tokenizer.json") as f:
            tj = json.load(f)
        for entry in tj.get("added_tokens", []):
            name_to_id[entry["content"]] = entry["id"]

    stop_ids = set()
    for c in ["<|END_OF_TURN_TOKEN|>", "<EOS_TOKEN>", "<EOP_TOKEN>",
              "<|START_OF_TURN_TOKEN|>", "<end_of_turn>", "</s>"]:
        if c in name_to_id:
            stop_ids.add(name_to_id[c])
    eos = cfg.get("eos_token_id")
    if eos:
        stop_ids.update(eos if isinstance(eos, list) else [eos])

    def first(*names):
        for n in names:
            if n in name_to_id: return name_to_id[n]
        return None

    return {
        "bos":      first("<BOS_TOKEN>") or cfg.get("bos_token_id", 2),
        "stop_ids": stop_ids,
        "start":    first("<|START_OF_TURN_TOKEN|>"),
        "end":      first("<|END_OF_TURN_TOKEN|>"),
        "user":     first("<|USER_TOKEN|>"),
        "system":   first("<|SYSTEM_TOKEN|>"),
        "chatbot":  first("<|CHATBOT_TOKEN|>"),
        # Extended tokens for thinking/tool models
        "start_response":    first("<|START_RESPONSE|>"),
        "end_response":      first("<|END_RESPONSE|>"),
        "start_action":      first("<|START_ACTION|>"),
        "end_action":        first("<|END_ACTION|>"),
        "start_tool_result": first("<|START_TOOL_RESULT|>"),
        "end_tool_result":   first("<|END_TOOL_RESULT|>"),
        "start_thinking":    first("<|START_THINKING|>"),
        "end_thinking":      first("<|END_THINKING|>"),
    }


# ── KV Cache ──────────────────────────────────────────────────────────────────

class CacheState:
    def __init__(self, n_layers):
        self.kv     = [[None, None] for _ in range(n_layers)]
        self.offset = 0

    def reset(self):
        self.kv     = [[None, None] for _ in range(len(self.kv))]
        self.offset = 0

    def feed_token(self, model, token_id, device, dtype):
        logits = model.forward_one(token_id, self.offset, self.kv, device, dtype)
        self.offset += 1
        return logits

    def feed_tokens(self, model, token_ids, device, dtype):
        logits = None
        for tid in token_ids:
            logits = self.feed_token(model, tid, device, dtype)
        return logits


# ── Generation ────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate(
    model, cache, new_user_ids, device, dtype, tokenizer, special,
    max_new_tokens=512, temperature=1.0, top_p=0.95, max_chars=100,
):
    logits    = cache.feed_tokens(model, new_user_ids, device, dtype)
    generated = []

    # Track section state for models with thinking/action tokens
    in_thinking = False
    thinking_ids = []

    for _ in range(max_new_tokens):
        if temperature > 0:
            probs  = F.softmax(logits / temperature, dim=-1)
            sp, si = torch.sort(probs, descending=True)
            cum    = torch.cumsum(sp, dim=-1)
            sp[(cum - sp) > top_p] = 0.0
            sp    /= sp.sum()
            next_id = si[0, torch.multinomial(sp[0], 1)].item()
        else:
            next_id = int(logits.argmax(-1))

        # Handle thinking tokens — swallow silently, show as prefix
        if special.get("start_thinking") and next_id == special["start_thinking"]:
            in_thinking = True
            logits = cache.feed_token(model, next_id, device, dtype)
            continue
        if special.get("end_thinking") and next_id == special["end_thinking"]:
            in_thinking = False
            if thinking_ids:
                t = tokenizer.decode(thinking_ids).strip()
                print(f"[thinking] {t[:100]}{'...' if len(t) > 100 else ''}")
                thinking_ids = []
            logits = cache.feed_token(model, next_id, device, dtype)
            continue
        if in_thinking:
            thinking_ids.append(next_id)
            logits = cache.feed_token(model, next_id, device, dtype)
            continue

        # Skip START_RESPONSE wrapper token if present
        if special.get("start_response") and next_id == special["start_response"]:
            logits = cache.feed_token(model, next_id, device, dtype)
            continue

        if next_id in special["stop_ids"]:
            cache.feed_token(model, next_id, device, dtype)
            break
        if special.get("end_response") and next_id == special["end_response"]:
            cache.feed_token(model, next_id, device, dtype)
            break

        generated.append(next_id)

        full  = tokenizer.decode(generated).strip()
        prev  = tokenizer.decode(generated[:-1]).strip() if len(generated) > 1 else ""
        chunk = re.sub(r'\s+', ' ', full[len(prev):])
        print(chunk, end="", flush=True)

        if len(full) >= max_chars:
            end_id = special.get("end") or next(iter(special["stop_ids"]), None)
            if end_id:
                cache.feed_token(model, end_id, device, dtype)
            break

        logits = cache.feed_token(model, next_id, device, dtype)
    else:
        end_id = special.get("end") or next(iter(special["stop_ids"]), None)
        if end_id:
            cache.feed_token(model, end_id, device, dtype)

    print()
    return AssistantResponse(content=tokenizer.decode(generated), max_chars=max_chars)


# ── Chat Log ──────────────────────────────────────────────────────────────────

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def new_log_path(log_dir):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / f"chat_log_{ts}.json"

def make_session(username, model_dir, cfg, args):
    arch = cfg.get("model_type", "cohere")
    return {
        "session": {
            "started_at": now_iso(),
            "username":   username,
            "model": {
                "name":                cfg.get("architectures", [arch])[0],
                "architecture":        arch,
                "model_dir":           str(Path(model_dir).resolve()),
                "num_hidden_layers":   cfg.get("num_hidden_layers"),
                "hidden_size":         cfg.get("hidden_size"),
                "num_attention_heads": cfg.get("num_attention_heads"),
                "vocab_size":          cfg.get("vocab_size"),
                "sliding_window":      cfg.get("sliding_window"),
                "num_experts":         cfg.get("num_experts"),
                "num_experts_per_tok": cfg.get("num_experts_per_tok"),
                "quantization":        "bfloat16" if args.quantize else "none",
            },
            "settings": {
                "dtype":          args.dtype,
                "max_new_tokens": args.max_new_tokens,
                "temperature":    args.temperature,
                "top_p":          args.top_p,
                "max_chars":      args.max_chars,
                "system_prompt":  args.system,
            },
        },
        "turns": [],
    }

def save_session(log_path, session):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(session, f, indent=2, ensure_ascii=False)


# ── Size sniffing for MoE ─────────────────────────────────────────────────────

def sniff_moe_sizes(model_dir, cfg):
    """
    Read actual tensor shapes from the first MoE shard to get correct
    dense_intermediate_size and expert_intermediate_size.
    Avoids hardcoding or mis-deriving these from config.
    """
    import copy
    cfg = copy.deepcopy(cfg)
    shards = sorted(Path(model_dir).glob("*.safetensors"))
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            # Dense layer 0 intermediate size
            for k in keys:
                if k == "model.layers.0.mlp.gate_proj.weight":
                    shape = f.get_tensor(k).shape
                    cfg["dense_intermediate_size"] = shape[0]
            # Expert intermediate size from any expert weight
            for k in keys:
                if ".mlp.experts.0.gate_proj.weight" in k:
                    shape = f.get_tensor(k).shape
                    cfg["expert_intermediate_size"] = shape[0]
            if "dense_intermediate_size" in cfg and "expert_intermediate_size" in cfg:
                break
    print(f"  Sniffed sizes: dense_inter={cfg.get('dense_intermediate_size')}, "
          f"expert_inter={cfg.get('expert_intermediate_size')}")
    return cfg


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",      required=True)
    parser.add_argument("--prompt",         default=None)
    parser.add_argument("--system",         default="You are a helpful assistant. Answer concisely.")
    parser.add_argument("--log-dir",        default="chat_logs")
    parser.add_argument("--username",       default=None)
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--top-p",          type=float, default=0.95)
    parser.add_argument("--max-chars",      type=int,   default=100,
                        help="Max characters in response (default: 100)")
    parser.add_argument("--quantize",       action="store_true")
    parser.add_argument("--dtype",          default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    log_dir  = Path(args.log_dir)
    username = args.username or getpass.getuser()
    device   = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    dtype    = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                "float32": torch.float32}[args.dtype]

    with open(Path(args.model_dir) / "config.json") as f:
        cfg = json.load(f)

    arch       = cfg.get("model_type", "cohere")
    model_type = arch
    print(f"Device  : {device}")
    print(f"User    : {username}")
    print(f"Model   : {arch}  ({cfg.get('num_hidden_layers')}L x {cfg.get('hidden_size')}d)")
    if cfg.get("sliding_window"):
        lt       = cfg.get("layer_types", [])
        n_slide  = sum(1 for t in lt if t == "sliding_attention")
        n_full   = sum(1 for t in lt if t == "full_attention")
        print(f"Attn    : {n_slide} sliding (window={cfg['sliding_window']}) + {n_full} full")
    if cfg.get("num_experts"):
        print(f"MoE     : {cfg['num_experts']} experts, {cfg.get('num_experts_per_tok',8)} active/token")

    # For cohere2_moe, sniff actual layer sizes from checkpoint before building model
    if cfg.get("model_type") == "cohere2_moe":
        cfg = sniff_moe_sizes(args.model_dir, cfg)

    print("Building model...")
    model = CohereModel(cfg).to(dtype)
    load_weights(model, args.model_dir, cfg)

    if args.quantize:
        print("Quantizing (bfloat16)...")
        quantize_model(model)

    model.to(device)
    model.eval()
    # Disable gradients for all parameters — inference only, saves ~2x memory
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = load_tokenizer(args.model_dir)
    special   = get_special_ids(args.model_dir, cfg)
    print(f"Special : bos={special['bos']}  stop_ids={special['stop_ids']}")
    n_layers  = cfg["num_hidden_layers"]

    def start_session():
        sess  = make_session(username, args.model_dir, cfg, args)
        lpath = new_log_path(log_dir)
        cache = CacheState(n_layers)
        init_ids = [special["bos"]]
        if args.system and special.get("start") and special.get("system"):
            init_ids += [special["start"], special["system"]]
            init_ids += tokenizer.encode(args.system).ids
            init_ids += [special["end"]]
        elif args.system:
            init_ids += tokenizer.encode(f"SYSTEM: {args.system}\n").ids
        # For models with start_response, open chatbot turn correctly
        cache.feed_tokens(model, init_ids, device, dtype)
        save_session(lpath, sess)
        print(f"Chat log: {lpath.resolve()}")
        return sess, lpath, cache

    session, log_path, cache = start_session()

    print(f"Response limit: {args.max_chars} chars")
    print("=" * 60)
    print(f"{arch}  |  'quit' to exit  |  'reset' for new session")
    print("=" * 60)

    while True:
        if args.prompt:
            user_input, args.prompt = args.prompt, None
        else:
            try:
                user_input = input(f"\n{username}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

        if user_input.lower() == "quit":   break
        if user_input.lower() == "reset":
            session, log_path, cache = start_session()
            print("New session started.")
            continue
        if not user_input: continue

        # Build turn — for models with start_response, open the response section
        if special.get("start") and special.get("user") and special.get("end") and special.get("chatbot"):
            new_ids = (
                [special["start"], special["user"]]
                + tokenizer.encode(user_input).ids
                + [special["end"], special["start"], special["chatbot"]]
            )
            # North-Mini-Code / Tiny-Aya wrap responses in START_RESPONSE
            if special.get("start_response"):
                new_ids += [special["start_response"]]
        else:
            new_ids = tokenizer.encode(f"USER: {user_input}\nASSISTANT:").ids

        user_ts = now_iso()
        print("\nModel: ", end="", flush=True)

        response = generate(
            model, cache, new_ids, device, dtype, tokenizer, special,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_chars=args.max_chars,
        )
        assistant_ts = now_iso()

        if response.truncated:
            print(f"  [truncated to {response.max_chars} chars]")

        session["turns"].append({
            "turn":      len(session["turns"]) + 1,
            "user":      {"username": username, "content": user_input, "timestamp": user_ts},
            "assistant": {"content": response.content, "char_count": response.char_count,
                          "truncated": response.truncated, "timestamp": assistant_ts},
        })
        save_session(log_path, session)


if __name__ == "__main__":
    main()
