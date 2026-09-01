#!/usr/bin/env python3
"""Build complete tiny valid models (llama / qwen3 / gemma4) with full
metadata and vocab, usable by Ember AND llama.cpp. Written into the
comparative fixtures dir; corpus entries are added by gen_corpus.py.

    python research/embersec/comparative/gen_valid_models.py
"""

from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[3]
GGUF = ROOT / "research" / "embersec" / "comparative" / "fixtures" / "gguf"
MAGIC = 0x46554747


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def i32(v):
    return struct.pack("<i", v)


def f32(v):
    return struct.pack("<f", v)


def s(b):
    if isinstance(b, str):
        b = b.encode()
    return u64(len(b)) + b


def pad(buf, align=32):
    while len(buf) % align:
        buf += b"\x00"
    return buf


def kv_str(buf, key, value):
    return buf + s(key) + u32(8) + s(value.encode())


def kv_u32(buf, key, value):
    return buf + s(key) + u32(4) + u32(value)


def kv_f32(buf, key, value):
    return buf + s(key) + u32(6) + f32(value)


def kv_array_str(buf, key, values):
    out = buf + s(key) + u32(9) + u32(8) + u64(len(values))
    for v in values:
        out += s(v.encode() if isinstance(v, str) else v)
    return out


def kv_array_f32(buf, key, values):
    out = buf + s(key) + u32(9) + u32(6) + u64(len(values))
    for v in values:
        out += f32(v)
    return out


def kv_array_i32(buf, key, values):
    out = buf + s(key) + u32(9) + u32(5) + u64(len(values))
    for v in values:
        out += i32(v)
    return out


def tensor_info(buf, name, dims, dtype, offset):
    out = buf + s(name) + u32(len(dims))
    for d in dims:
        out += u64(d)
    return out + u32(dtype) + u64(offset)


def build_model(arch, hparams, tensors, vocab):
    """tensors: list of (name, dims, dtype, payload). vocab: dict with
    tokens/scores/token_types/special ids."""
    n = len(tensors)
    kv_count = 3 + len(hparams) + 6 + 3  # arch/name/file_type + hparams + model/pre/merges + vocab arrays + ids
    buf = u32(MAGIC) + u32(3) + u64(n) + u64(kv_count)
    buf = kv_str(buf, "general.architecture", arch)
    buf = kv_str(buf, "general.name", f"tiny-{arch}-control")
    buf = kv_u32(buf, "general.file_type", 1)  # f32
    for key, value in hparams:
        if isinstance(value, float):
            buf = kv_f32(buf, key, value)
        else:
            buf = kv_u32(buf, key, value)
    # "gpt2" = BPE vocab type (matches real Llama-3 GGUFs); "llama" would
    # select the SPM type whose byte_to_token looks up <0xNN>/raw-byte
    # tokens instead of the gpt2 byte encoder.
    buf = kv_str(buf, "tokenizer.ggml.model", "gpt2")
    buf = kv_str(buf, "tokenizer.ggml.pre", "llama-bpe")
    buf = kv_array_str(buf, "tokenizer.ggml.merges", [])
    buf = kv_array_str(buf, "tokenizer.ggml.tokens", vocab["tokens"])
    buf = kv_array_f32(buf, "tokenizer.ggml.scores", vocab["scores"])
    buf = kv_array_i32(buf, "tokenizer.ggml.token_type", vocab["token_type"])
    for key in ("bos", "eos", "unk"):
        buf = kv_u32(buf, f"tokenizer.ggml.{key}_token_id", vocab[f"{key}_id"])
    offset = 0
    for name, dims, dtype, payload in tensors:
        buf = tensor_info(buf, name, dims, dtype, offset)
        # llama.cpp requires 32-byte-aligned tensor data offsets
        offset += len(payload) + (-len(payload)) % 32
    buf = pad(buf)
    for name, dims, dtype, payload in tensors:
        buf += payload
        while len(buf) % 32:
            buf += b"\x00"
    return buf


VOCAB8 = {
    # multi-character words only: single-char entries would collide with
    # the byte-fallback tokens added by vocab_with_bytes
    "tokens": ["<unk>", "<s>", "</s>", "<pad>", "hello", "world", "foo", "bar"],
    "scores": [0.0] * 8,
    "token_type": [2, 3, 3, 3, 1, 1, 1, 1],
    "unk_id": 0, "bos_id": 1, "eos_id": 2,
}


def vocab_with_bytes(base_vocab):
    """llama.cpp BPE requires the 256 byte tokens in the vocab, keyed by
    the GPT-2 byte encoder (llama.cpp unicode.cpp unicode_byte_to_utf8):
    0x21-0x7E -> U+0021-U+007E, 0xA1-0xAC -> U+00A1-U+00AC,
    0xAE-0xFF -> U+00AE-U+00FF, and every other byte -> U+0100+n in order
    (0x00 -> U+0100 '\u0100', 0x0A -> U+010A, 0x20 -> U+0120, ...).
    Raw NUL must not appear in a token text (C-string handling in
    gguf_get_arr_str truncates it)."""
    b_enc = {}
    b_enc.update({b: b for b in range(0x21, 0x7F)})
    b_enc.update({b: b for b in range(0xA1, 0xAD)})
    b_enc.update({b: b for b in range(0xAE, 0x100)})
    n = 0
    for b in range(0x100):
        if b not in b_enc:
            b_enc[b] = 0x100 + n
            n += 1
    tokens = list(base_vocab["tokens"])
    token_type = list(base_vocab["token_type"])
    for b in range(0x100):
        tokens.append(chr(b_enc[b]).encode("utf-8"))
        token_type.append(1)
    scores = [0.0] * len(tokens)
    return {
        "tokens": tokens,
        "scores": scores,
        "token_type": token_type,
        "unk_id": base_vocab["unk_id"],
        "bos_id": base_vocab["bos_id"],
        "eos_id": base_vocab["eos_id"],
    }


def f32_tensor(dims, value=0.01):
    n = 1
    for d in dims:
        n *= d
    return dims, 0, struct.pack(f"<{n}f", *([value] * n))


def llama_tensors():
    t = []
    t.append(("token_embd.weight",) + f32_tensor([8, 8], 0.01))
    t.append(("output_norm.weight",) + f32_tensor([8], 1.0))
    t.append(("output.weight",) + f32_tensor([8, 8], 0.01))
    for name, dims in [
        ("attn_q.weight", [8, 8]), ("attn_k.weight", [8, 4]),
        ("attn_v.weight", [8, 4]), ("attn_output.weight", [8, 8]),
        ("ffn_gate.weight", [8, 16]), ("ffn_up.weight", [8, 16]),
        ("ffn_down.weight", [16, 8]),
    ]:
        t.append((f"blk.0.{name}",) + f32_tensor(dims, 0.01))
    for name in ["attn_norm.weight", "ffn_norm.weight"]:
        t.append((f"blk.0.{name}",) + f32_tensor([8], 1.0))
    return t


def main():
    GGUF.mkdir(parents=True, exist_ok=True)

    llama_hp = [
        ("llama.block_count", 1), ("llama.attention.head_count", 2),
        ("llama.attention.head_count_kv", 1), ("llama.embedding_length", 8),
        ("llama.attention.key_length", 4), ("llama.context_length", 8),
        ("llama.vocab_size", 8), ("llama.feed_forward_length", 16),
        ("llama.attention.layer_norm_rms_epsilon", 1e-5),
        ("llama.rope.freq_base", 10000.0),
    ]
    vocab264 = vocab_with_bytes(VOCAB8)
    v = len(vocab264["tokens"])
    llama_t = llama_tensors()
    llama_t[0] = ("token_embd.weight",) + f32_tensor([8, v], 0.01)
    llama_t[2] = ("output.weight",) + f32_tensor([8, v], 0.01)
    llama_hp_full = [(k, v if k == "llama.vocab_size" else x) for k, x in llama_hp]
    (GGUF / "tiny_llama_full.bin").write_bytes(
        build_model("llama", llama_hp_full, llama_t, vocab264))

    qwen3_hp = [
        ("qwen3.block_count", 1), ("qwen3.attention.head_count", 2),
        ("qwen3.attention.head_count_kv", 1), ("qwen3.embedding_length", 8),
        ("qwen3.attention.key_length", 4), ("qwen3.context_length", 8),
        ("qwen3.vocab_size", 8), ("qwen3.feed_forward_length", 16),
        ("qwen3.attention.layer_norm_rms_epsilon", 1e-5),
        ("qwen3.rope.freq_base", 10000.0),
    ]
    q3 = llama_tensors()
    q3[0] = ("token_embd.weight",) + f32_tensor([8, v], 0.01)
    q3[2] = ("output.weight",) + f32_tensor([8, v], 0.01)
    # qwen3 uses qk norms
    for name in ["attn_q_norm.weight", "attn_k_norm.weight"]:
        q3.append((f"blk.0.{name}",) + f32_tensor([4], 1.0))
    qwen3_hp_full = [(k, v if k == "qwen3.vocab_size" else x) for k, x in qwen3_hp]
    (GGUF / "tiny_qwen3_full.bin").write_bytes(
        build_model("qwen3", qwen3_hp_full, q3, vocab264))

    gemma4_hp = [
        ("gemma4.block_count", 1), ("gemma4.embedding_length", 2),
        ("gemma4.attention.head_count", 1),
        ("gemma4.attention.head_count_kv", 1),
        ("gemma4.attention.key_length", 2), ("gemma4.feed_forward_length", 2),
        ("gemma4.vocab_size", 4), ("gemma4.context_length", 8),
        ("gemma4.attention.sliding_window", 2),
        ("gemma4.attention.layer_norm_rms_epsilon", 1e-6),
        ("gemma4.rope.freq_base", 1000000.0),
        ("gemma4.rope.freq_base_swa", 10000.0),
        ("gemma4.attention.scale", 8.0),
    ]
    g4 = []
    g4.append(("token_embd.weight",) + f32_tensor([2, 4], 0.1))
    g4.append(("output_norm.weight",) + f32_tensor([2], 1.0))
    g4.append(("output.weight",) + f32_tensor([2, 4], 0.1))
    for name in ["attn_q", "attn_k", "attn_v", "attn_output",
                 "ffn_gate", "ffn_up", "ffn_down"]:
        g4.append((f"blk.0.{name}.weight",) + f32_tensor([2, 2], 0.01))
    for name in ["attn_q_norm", "attn_k_norm", "attn_norm", "attn_post_norm",
                 "ffn_norm", "ffn_post_norm"]:
        g4.append((f"blk.0.{name}.weight",) + f32_tensor([2], 1.0))
    vocab4 = {
        "tokens": ["<unk>", "<s>", "</s>", "a"],
        "scores": [0.0] * 4,
        "token_type": [2, 3, 3, 1],
        "unk_id": 0, "bos_id": 1, "eos_id": 2,
    }
    (GGUF / "tiny_gemma4_full.bin").write_bytes(
        build_model("gemma4", gemma4_hp, g4, vocab4))
    print("wrote tiny_llama_full.bin, tiny_qwen3_full.bin, tiny_gemma4_full.bin")


if __name__ == "__main__":
    main()
