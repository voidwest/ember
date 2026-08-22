#!/usr/bin/env python3
"""Sub-operation forensics for all 35 Gemma 4 blocks.

Path A ("hf"): the validated fp32 transformers model stepped manually
  (mirroring modeling_gemma4.py) with every intermediate dumped.
Path B ("gguf"): identical math, but every weight taken from the Q8_0/BF16
  GGUF via dequantization and structured exactly like src/gemma4.rs
  (same names, same order, same scales).

A-vs-B isolates *algorithm* differences from weight-quantization noise:
both run fp32 on the same activations, so a materially divergent stage is
a real ember-side misinterpretation, not floating-point drift.
"""
import json
import os
import sys

import numpy as np
import torch

SRC = "/home/west/ember-work/gemma4-src"
GGUF = "/home/west/ember/models/gemma-4-E2B-it.Q8_0.gguf"
IDS = [2, 9259]  # BOS + "Hello" (matches the ember --dump-layers run)
EPS = 1e-6


# ------------------------------------------------------------------ gguf i/o


def dequant_q8_0(raw: bytes, n_elements: int) -> np.ndarray:
    n_blocks = n_elements // 32
    buf = np.frombuffer(raw, dtype=np.uint8)[: n_blocks * 34].reshape(n_blocks, 34)
    scales = buf[:, :2].copy().view("<f2").astype(np.float32)
    qs = buf[:, 2:].copy().view(np.int8).astype(np.float32)
    return (scales * qs).reshape(-1)[:n_elements]


class Gguf:
    def __init__(self, path):
        import gguf as gguf_lib

        reader = gguf_lib.GGUFReader(path)
        self.tensors = {t.name: t for t in reader.tensors}

    def rows_bytes(self, name: str, row_idx):
        """Raw bytes of selected rows (row = outermost hf dim)."""
        t = self.tensors[name]
        shape = tuple(int(d) for d in reversed(t.shape))
        inner = int(np.prod(shape[1:]))
        tt = int(t.tensor_type)
        if tt == 8:
            row_bytes = (inner // 32) * 34
        elif tt == 30:
            row_bytes = inner * 2
        else:
            row_bytes = inner * 4
        base = np.frombuffer(bytes(t.data), dtype=np.uint8)
        return base[row_idx[0] * row_bytes : (row_idx[-1] + 1) * row_bytes], shape, tt

    def get_rows(self, name: str, row_idx) -> torch.Tensor:
        """Dequantize only the selected rows."""
        t = self.tensors[name]
        shape = tuple(int(d) for d in reversed(t.shape))
        inner = int(np.prod(shape[1:]))
        idx = sorted(row_idx)
        raw, _, tt = self.rows_bytes(name, idx)
        # sliced-buffer offsets are relative to the FIRST selected row
        first = idx[0]
        offsets = {r: r - first for r in row_idx}
        if tt == 8:
            row_bytes = (inner // 32) * 34
            arrs = []
            for r in row_idx:
                seg = raw[offsets[r] * row_bytes : (offsets[r] + 1) * row_bytes]
                arrs.append(dequant_q8_0(seg, inner))
            arr = np.stack(arrs)
            return torch.from_numpy(arr)
        if tt == 30:
            allb = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(
                np.float32
            ).reshape(-1, inner)
            return torch.from_numpy(allb[[offsets[r] for r in row_idx]].copy())
        raise ValueError("only q8_0/bf16 supported here")

    def get(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        shape = tuple(int(d) for d in reversed(t.shape))  # hf shape
        n = int(np.prod(shape))
        code = int(t.tensor_type) if str(t.tensor_type).isdigit() else {
            "Q8_0": 8, "BF16": 30, "F32": 0,
        }.get(str(t.tensor_type).split(".")[-1], -1)
        raw = bytes(t.data)
        if code == 8:
            arr = dequant_q8_0(raw, n)
        elif code == 30:
            arr = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(
                np.float32
            )
        elif code == 0:
            arr = np.frombuffer(raw, dtype=np.float32)
        else:
            raise ValueError(f"unhandled gguf dtype {t.tensor_type} for {name}")
        return torch.from_numpy(arr.reshape(shape).copy())


# ------------------------------------------------------------------- helpers


def rms_norm(x, w, eps=EPS):
    v = x.float().pow(2).mean(-1, keepdim=True)
    out = x.float() * torch.rsqrt(v + eps)
    return out if w is None else out * w.float()


def gelu_tanh(x):
    return torch.nn.functional.gelu(x, approximate="tanh")


def rope_half(x, pos_start, inv_freq, factors=None):
    """x [seq, heads, head_dim]; NEOX half-split on first 2*half dims."""
    seq, n_heads, head_dim = x.shape
    half = inv_freq.shape[0]
    theta = (
        torch.arange(pos_start, pos_start + seq, dtype=torch.float32)[:, None, None]
        * inv_freq[None, None, :half]
    )
    if factors is not None:
        theta = theta / factors[:half].clamp(min=1.0)[None, None, :]
    cos, sin = theta.cos(), theta.sin()
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    out = torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos, x[..., 2 * half :]], dim=-1
    )
    return out


def attention(q, k, v, scale, sliding_window=None, total_len=None):
    """q/k/v: [heads, seq, hd] (k/v already expanded). Returns [heads, seq, hd]."""
    seq = q.shape[1]
    scores = q.float() @ k.float().transpose(-1, -2) * scale
    if total_len is None:
        total_len = seq
    pos_offsets = torch.arange(total_len - seq, total_len).unsqueeze(1)
    key_pos = torch.arange(total_len).unsqueeze(0)
    causal = key_pos <= pos_offsets
    if sliding_window is not None:
        causal &= key_pos > pos_offsets - sliding_window
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return probs.to(v.dtype) @ v.float(), probs


# ---------------------------------------------------------------------- main


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/home/west/ember-work/gemma_forensics"
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextConfig,
        Gemma4TextModel,
        ROPE_INIT_FUNCTIONS,
    )

    tok = AutoTokenizer.from_pretrained(SRC)
    cfg_full = json.load(open(os.path.join(SRC, "config.json")))
    cfg = Gemma4TextConfig(**cfg_full["text_config"])

    ids = IDS
    seq = len(ids)
    input_ids = torch.tensor([ids])

    with torch.device("meta"):
        model = Gemma4TextModel(cfg)
    model.rotary_emb = model.rotary_emb.to_empty(device="cpu")
    for lt in set(cfg.layer_types):
        rp = cfg.rope_parameters[lt]
        fn = (
            model.rotary_emb.compute_default_rope_parameters
            if rp["rope_type"] == "default"
            else ROPE_INIT_FUNCTIONS[rp["rope_type"]]
        )
        kw = {"device": torch.device("cpu"), "layer_type": lt}
        if lt == "full_attention" and rp["rope_type"] == "proportional":
            kw["head_dim_key"] = "global_head_dim"
        inv_freq, scaling = fn(cfg, **kw)
        model.rotary_emb.register_buffer(f"{lt}_inv_freq", inv_freq, persistent=False)
    model.eval()

    st_path = os.path.join(SRC, "model.safetensors")
    from safetensors import safe_open

    sf = safe_open(st_path, framework="pt")

    def hf_tensor(name):
        return sf.get_tensor(f"model.language_model.{name}")

    uniq = sorted(set(ids))
    remap = {t: i for i, t in enumerate(uniq)}

    embed_rows = hf_tensor("embed_tokens.weight")[uniq].float()
    ple_rows = hf_tensor("embed_tokens_per_layer.weight")[uniq]
    assign_into = {}

    def assign(root, path, tensor):
        parts = path.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        old = getattr(parent, leaf)
        if isinstance(old, torch.nn.Parameter):
            setattr(parent, leaf, torch.nn.Parameter(tensor, requires_grad=False))
        else:
            setattr(parent, leaf, tensor)

    with torch.no_grad():
        assign(model.norm, "weight", hf_tensor("norm.weight").float())
        assign(model, "per_layer_model_projection.weight",
               hf_tensor("per_layer_model_projection.weight").float())
        assign(model, "per_layer_projection_norm.weight",
               hf_tensor("per_layer_projection_norm.weight").float())
        names = [
            "input_layernorm.weight", "self_attn.q_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_proj.weight",
            "self_attn.k_norm.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight", "mlp.gate_proj.weight",
            "mlp.up_proj.weight", "mlp.down_proj.weight",
            "post_feedforward_layernorm.weight", "per_layer_input_gate.weight",
            "per_layer_projection.weight", "post_per_layer_input_norm.weight",
            "layer_scalar",
        ]
        for i in range(cfg.num_hidden_layers):
            layer = model.layers[i]
            for name in names:
                try:
                    t = hf_tensor(f"layers.{i}.{name}")
                except Exception:
                    continue
                probe = layer
                try:
                    for part in name.split(".")[:-1]:
                        probe = getattr(probe, part)
                except AttributeError:
                    continue
                assign(layer, name, t.float() if t.is_floating_point() else t)

    # ---- shared inputs ---------------------------------------------------
    embed_scale = cfg.hidden_size ** 0.5
    emb_hf = embed_rows[[remap[t] for t in ids]].unsqueeze(0) * embed_scale  # [1,seq,H]

    ple_scale = cfg.hidden_size_per_layer_input ** 0.5
    ple_tok = ple_rows[[remap[t] for t in ids]].float().view(seq, 35, 256) * ple_scale
    ctx = model.per_layer_model_projection(emb_hf[0]) * (cfg.hidden_size ** -0.5)
    ctx = ctx.view(seq, 35, 256)
    ctx = rms_norm(ctx, model.per_layer_projection_norm.weight)
    ple_combined = (ctx + ple_tok) * (2 ** -0.5)  # [seq, 35, 256]

    inv_freq = {lt: getattr(model.rotary_emb, f"{lt}_inv_freq").float()
                for lt in set(cfg.layer_types)}

    # ---- path A: step the fp32 model manually ----------------------------
    dumps_a = {}
    kv_store = {}
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers
    x = emb_hf.clone()
    with torch.no_grad():
        for li in range(35):
            layer = model.layers[li]
            lt = cfg.layer_types[li]
            p = f"L{li}."
            residual = x
            normed = rms_norm(x, layer.input_layernorm.weight)
            if li < 6:
                dumps_a[p + "attn_norm"] = normed.clone()

            attn = layer.self_attn
            head_dim = cfg.global_head_dim if lt == "full_attention" else cfg.head_dim
            q = attn.q_proj(normed[0]).view(seq, 8, head_dim)
            q = rms_norm(q, attn.q_norm.weight)
            q = rope_half(q, 0, inv_freq[lt]).transpose(0, 1)  # [h, s, d]
            is_shared = li >= first_shared > 0
            if not is_shared:
                k = attn.k_proj(normed[0]).view(seq, 1, head_dim)
                k = rms_norm(k, attn.k_norm.weight)
                k = rope_half(k, 0, inv_freq[lt]).transpose(0, 1)
                v = attn.v_proj(normed[0]).view(seq, 1, head_dim)
                v = rms_norm(v, None)  # v_norm has no scale
                v = v.transpose(0, 1)
                kv_store[lt] = (k, v)
            else:
                k, v = kv_store[lt]

            ao, probs = attention(
                q, k, v, 1.0,
                sliding_window=cfg.sliding_window if lt == "sliding_attention" else None,
            )
            dumps_a[p + "attn_out"] = ao.transpose(0, 1).clone()
            o = attn.o_proj(ao.transpose(0, 1).reshape(seq, -1))
            dumps_a[p + "o_proj"] = o.clone()
            x = residual + rms_norm(o, layer.post_attention_layernorm.weight)
            dumps_a[p + "post_attn_add"] = x.clone()

            residual = x
            h = rms_norm(x, layer.pre_feedforward_layernorm.weight)
            mlp = layer.mlp.down_proj(
                gelu_tanh(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h)
            )
            dumps_a[p + "ffn"] = mlp.clone()
            x = residual + rms_norm(mlp, layer.post_feedforward_layernorm.weight)
            dumps_a[p + "ffn_add"] = x.clone()

            residual = x
            g = layer.per_layer_input_gate(x[0])
            g = gelu_tanh(g)
            g = g * ple_combined[:, li, :]
            pp = layer.per_layer_projection(g)
            x = residual + rms_norm(pp, layer.post_per_layer_input_norm.weight)
            dumps_a[p + "ple_add"] = x.clone()

            x = x * layer.layer_scalar
            dumps_a[p + "block_out"] = x.clone()

    # ---- path B: same math, GGUF-dequantized weights ---------------------
    g = Gguf(GGUF)
    W = lambda n: g.get(n)

    dumps_b = {}
    xb = (
        g.get_rows("token_embd.weight", ids).float() * embed_scale
    ).unsqueeze(0)

    # per-layer input pathway from GGUF tables
    ple_tok_b = (
        g.get_rows("per_layer_token_embd.weight", ids).float().view(seq, 35, 256)
        * ple_scale
    )
    proj_w = W("per_layer_model_proj.weight")  # hf [8960, 1536]? verify below
    if proj_w.shape != (35 * 256, cfg.hidden_size):
        proj_w = proj_w.t()
    ctxb = (xb[0] @ proj_w.t()) * (cfg.hidden_size ** -0.5)
    ctxb = ctxb.view(seq, 35, 256)
    proj_norm_w = W("per_layer_proj_norm.weight")
    ctxb = rms_norm(ctxb, proj_norm_w)
    ple_b = (ctxb + ple_tok_b) * (2 ** -0.5)

    freqs_gguf = W("rope_freqs.weight")  # [256]
    kv_store_b = {}
    first_shared = cfg.num_hidden_layers - cfg.num_kv_shared_layers

    with torch.no_grad():
        for li in range(35):
            lt = cfg.layer_types[li]
            bp = f"blk.{li}."
            p = f"L{li}."
            head_dim = cfg.global_head_dim if lt == "full_attention" else cfg.head_dim

            def w(name, hf_shape=None):
                t = W(bp + name)
                return t

            def lin(name, xin):
                wt = W(bp + name)  # hf row-major [out, in]
                assert wt.shape[1] == xin.shape[-1], (name, wt.shape, xin.shape)
                return xin @ wt.t()

            residual = xb
            normed = rms_norm(xb, W(bp + "attn_norm.weight"))
            dumps_b[p + "attn_norm"] = normed.clone()

            q = lin("attn_q.weight", normed[0]).view(seq, 8, head_dim)
            q = rms_norm(q, W(bp + "attn_q_norm.weight"))
            dumps_b[p + "q_normed"] = q.clone()
            factors = freqs_gguf if False else None
            if lt == "full_attention":
                factors = freqs_gguf  # partial rotary: 64 pairs rotate
            q = rope_half(q, 0, inv_freq[lt], factors).transpose(0, 1)
            if li >= first_shared > 0:
                k, v = kv_store_b[lt]
            else:
                k = lin("attn_k.weight", normed[0]).view(seq, 1, head_dim)
                k = rms_norm(k, W(bp + "attn_k_norm.weight"))
                k = rope_half(k, 0, inv_freq[lt], factors).transpose(0, 1)
                v = lin("attn_v.weight", normed[0]).view(seq, 1, head_dim)
                v = rms_norm(v, None)
                v = v.transpose(0, 1)
                kv_store_b[lt] = (k, v)

            ao, _ = attention(
                q, k, v, 1.0,
                sliding_window=512 if lt == "sliding_attention" else None,
            )
            dumps_b[p + "attn_out"] = ao.transpose(0, 1).clone()
            o = lin("attn_output.weight", ao.transpose(0, 1).reshape(seq, -1))
            dumps_b[p + "o_proj"] = o.clone()
            xb = residual + rms_norm(o, W(bp + "post_attention_norm.weight"))
            dumps_b[p + "post_attn_add"] = xb.clone()

            residual = xb
            h = rms_norm(xb, W(bp + "ffn_norm.weight"))
            mlp = (
                lin("ffn_down.weight", gelu_tanh(lin("ffn_gate.weight", h)) * lin("ffn_up.weight", h))
            )
            dumps_b[p + "ffn"] = mlp.clone()
            xb = residual + rms_norm(mlp, W(bp + "post_ffw_norm.weight"))
            dumps_b[p + "ffn_add"] = xb.clone()

            residual = xb
            gate_w = W(bp + "inp_gate.weight")
            gl_in = xb[0] @ gate_w.t()
            dumps_b[p + "ple_gate_in"] = gl_in.clone()
            gl = gelu_tanh(gl_in)
            dumps_b[p + "ple_gelu"] = gl.clone()
            gl = gl * ple_b[:, li, :]
            dumps_b[p + "ple_mul"] = gl.clone()
            prj = W(bp + "proj.weight")
            assert prj.shape[1] == 256, prj.shape
            pp = gl @ prj.t()
            dumps_b[p + "ple_proj"] = pp.clone()
            pn = rms_norm(pp, W(bp + "post_norm.weight"))
            dumps_b[p + "ple_norm"] = pn.clone()
            xb = residual + pn
            dumps_b[p + "ple_add"] = xb.clone()

            xb = xb * W(bp + "layer_output_scale.weight")[0]
            dumps_b[p + "block_out"] = xb.clone()

    # persist path-B tensors for comparison against ember's own dumps
    for key, t in dumps_b.items():
        i = int(key[1 : key.index(".")])
        stage = key.split(".", 1)[1]
        np.save(
            os.path.join(out_dir, f"L{i}_{stage}.f32.npy"),
            t.detach().numpy().astype(np.float32),
        )

    # ---- compare ----------------------------------------------------------
    print(f"{'stage':22s} {'cos(A,B)':>10s} {'max_abs':>10s} {'rmsA':>8s} {'rmsB':>8s}")
    worst = []
    for key in dumps_a:
        a, b = dumps_a[key], dumps_b[key]
        af, bf = a.flatten().float(), b.flatten().float()
        cos = float(af @ bf / (af.norm() * bf.norm() + 1e-30))
        maxabs = float((af - bf).abs().max())
        ra, rb = float(af.pow(2).mean().sqrt()), float(bf.pow(2).mean().sqrt())
        flag = "" if cos > 0.995 else "   <-- DIVERGES"
        print(f"{key:22s} {cos:10.6f} {maxabs:10.4f} {ra:8.3f} {rb:8.3f}{flag}")
        if cos <= 0.995:
            worst.append(key)
    print("\ndivergent stages:", worst or "none")


if __name__ == "__main__":
    main()
