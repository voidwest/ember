#!/usr/bin/env python3
"""Generate the EmberSEC comparative corpus: fixtures + corpus.json.

Sources:
- fuzz/corpus seeds (copied, hashes recomputed here)
- exact minimized fuzz reproducers reconstructed from session records
  (tokenizer; origin marked "fuzz-discovered minimized artifact (reconstructed)")
- canonical synthetic fixtures for fuzz-discovered classes whose minimized
  bytes were not retained (origin marked "fuzz-discovered class; canonical
  synthetic fixture")
- structured synthetic boundary cases

Synthetic writes happen BEFORE the copy loop and before add() snapshots, so
corpus.json hashes always match the bytes on disk. All fixtures < 4 KiB.

    python research/embersec/comparative/gen_corpus.py
"""

from pathlib import Path
import hashlib
import json
import shutil
import struct

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research" / "embersec" / "comparative"
FIX = OUT / "fixtures"
CORPUS = OUT / "corpus.json"

GGUF = FIX / "gguf"
TOK = FIX / "tokenizer"

MAGIC = 0x46554747


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def u32(v):
    return struct.pack("<I", v)


def u64(v):
    return struct.pack("<Q", v)


def s(b):
    return u64(len(b)) + b


def pad(buf, align=32):
    while len(buf) % align:
        buf += b"\x00"
    return buf


def tensor_info(name, dims, dtype, offset):
    out = s(name) + u32(len(dims))
    for d in dims:
        out += u64(d)
    return out + u32(dtype) + u64(offset)


def single_tensor_gguf(name, dims, dtype, offset, payload, align=32):
    buf = u32(MAGIC) + u32(3) + u64(1) + u64(0)
    buf += tensor_info(name, dims, dtype, offset)
    buf = pad(buf, align)
    buf += b"\x00" * offset + payload
    return buf


def range_overflow_gguf():
    """offset chosen so data_start + offset fits but + byte_len wraps u64."""
    prefix = u32(MAGIC) + u32(3) + u64(1) + u64(0)
    probe = prefix + tensor_info(b"t.weight", [4], 0, 0)
    data_start = len(pad(bytearray(probe)))
    offset = (2**64 - 1) - 15 - data_start
    buf = prefix + tensor_info(b"t.weight", [4], 0, offset)
    return pad(buf)


def tiny_llama_gguf(overrides=None, one_d_attn_q=False):
    """One-layer llama GGUF: embed 8, heads 2/1 kv, head_dim 4, vocab 8,
    ffn 16, context 8. `overrides` maps metadata key -> u32 value."""
    overrides = overrides or {}
    tensors = [
        (b"token_embd.weight", [8, 8]),
        (b"output_norm.weight", [8]),
        (b"output.weight", [8, 8]),
        (b"blk.0.attn_q.weight", [8] if one_d_attn_q else [8, 8]),
        (b"blk.0.attn_k.weight", [8, 4]),
        (b"blk.0.attn_v.weight", [8, 4]),
        (b"blk.0.attn_output.weight", [8, 8]),
        (b"blk.0.ffn_gate.weight", [8, 16]),
        (b"blk.0.ffn_up.weight", [8, 16]),
        (b"blk.0.ffn_down.weight", [16, 8]),
        (b"blk.0.attn_norm.weight", [8]),
        (b"blk.0.ffn_norm.weight", [8]),
    ]
    buf = u32(MAGIC) + u32(3) + u64(len(tensors)) + u64(8)
    kvs = [
        (b"general.architecture", None),
        (b"llama.block_count", 1),
        (b"llama.attention.head_count", 2),
        (b"llama.attention.head_count_kv", 1),
        (b"llama.embedding_length", 8),
        (b"llama.attention.key_length", 4),
        (b"llama.context_length", 8),
        (b"llama.vocab_size", 8),
    ]
    for key, default in kvs:
        buf += s(key)
        value = overrides.get(key.decode(), default)
        if value is None:
            buf += u32(8) + s(b"llama")
        else:
            buf += u32(4) + u32(value)
    offset = 0
    for name, dims in tensors:
        buf += tensor_info(name, dims, 0, offset)
        offset += 4 * (dims[0] if len(dims) == 1 else dims[0] * dims[1])
    buf = pad(buf)
    for name, dims in tensors:
        n = dims[0] if len(dims) == 1 else dims[0] * dims[1]
        buf += b"\x00" * (4 * n)
    return buf


MINI_BPE = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [],
    "normalizer": None,
    "pre_tokenizer": None,
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "byte_fallback": False,
        "vocab": {"a": 0, "b": 1, "ab": 2, "hello": 3, "world": 4},
        "merges": [],
    },
}


def main():
    for d in (GGUF, TOK):
        d.mkdir(parents=True, exist_ok=True)

    # ---- synthetic fixtures (written BEFORE the copy loop and add()) ----
    (GGUF / "range_overflow.bin").write_bytes(range_overflow_gguf())
    (GGUF / "huge_dim_past_eof.bin").write_bytes(
        single_tensor_gguf(b"t.weight", [1 << 40], 0, 0, b""))
    (GGUF / "count_above_abs_cap.bin").write_bytes(
        u32(MAGIC) + u32(3) + u64(1_000_001) + u64(0))
    (GGUF / "tiny_llama_valid.bin").write_bytes(tiny_llama_gguf())
    (GGUF / "tiny_llama_hostile_ctx.bin").write_bytes(
        tiny_llama_gguf({"llama.context_length": 2**32 - 1}))
    (GGUF / "tiny_llama_hostile_layers.bin").write_bytes(
        tiny_llama_gguf({"llama.block_count": 1_000_000}))
    (GGUF / "tiny_llama_missing_tensors.bin").write_bytes(
        u32(MAGIC) + u32(3) + u64(0) + u64(2)
        + s(b"general.architecture") + u32(8) + s(b"llama")
        + s(b"llama.block_count") + u32(4) + u32(1))
    (GGUF / "tiny_llama_odd_keylen.bin").write_bytes(
        tiny_llama_gguf({"llama.attention.key_length": 1}))
    (GGUF / "tiny_llama_1d_attn_q.bin").write_bytes(
        tiny_llama_gguf(one_d_attn_q=True))
    (GGUF / "tiny_llama_rope_product.bin").write_bytes(tiny_llama_gguf({
        "llama.context_length": 16 * 1024 * 1024,
        "llama.attention.key_length": 4096,
    }))
    (GGUF / "tiny_llama_arch_gpt5.bin").write_bytes(
        u32(MAGIC) + u32(3) + u64(0) + u64(1)
        + s(b"general.architecture") + u32(8) + s(b"gpt5"))
    (GGUF / "tiny_llama_hostile_vocab.bin").write_bytes(
        tiny_llama_gguf({"llama.vocab_size": 5_000_000}))
    # q8_0 layout case: element-count-aligned but dims[0]-misaligned with a
    # FULL payload so the range check passes and the layout rule is what
    # fires. Baseline rejects later at the compressed constructor; current
    # rejects at the validation gate.
    (GGUF / "q8_0_dim_misaligned").write_bytes(
        single_tensor_gguf(b"t.weight", [16, 64], 8, 0, b"\x00" * (32 * 34)))
    # K-quant layout case: dims[0]=128 not 256-aligned, full 288-byte
    # payload. BASELINE eager path ACCEPTS this (element count 512 is
    # 256-aligned) and dequantizes with a non-compliant block order;
    # current rejects at the gate. Headline layout-gap delta.
    (GGUF / "q4_k_dim_misaligned").write_bytes(
        single_tensor_gguf(b"t.weight", [128, 4], 12, 0, b"\x00" * (2 * 144)))
    # tokenizer: exact minimized fuzz reproducers + canonical synthetic
    (TOK / "fuzz_decoder_invalid_utf8_26b.bin").write_bytes(
        b"\r\r{ \"decoder\"   : \r\r\r\r\x90\r\r ")
    (TOK / "fuzz_decoder_bad_value_15b.bin").write_bytes(b"\r{ \"decoder\":D\x89")
    (TOK / "synth_decoder_invalid_utf8_nested.bin").write_bytes(
        b'{\n"decoder"   : {\n"deco\x9ber" : 0\n}\n}')
    import json as _json
    (TOK / "valid_mini_bpe.json").write_bytes(_json.dumps(MINI_BPE).encode())

    # ---- copy fuzz corpus seeds (kept where they do not collide) --------
    for f in sorted((ROOT / "fuzz" / "corpus" / "gguf_loader").iterdir()):
        if f.is_file() and not (GGUF / f.name).exists():
            shutil.copy(f, GGUF / f.name)
    for f in sorted((ROOT / "fuzz" / "corpus" / "gguf_to_llama").iterdir()):
        if f.is_file() and not (GGUF / f.name).exists():
            shutil.copy(f, GGUF / f.name)
    for f in sorted((ROOT / "fuzz" / "corpus" / "tokenizer_json").iterdir()):
        if f.is_file() and not (TOK / f.name).exists():
            shutil.copy(f, TOK / f.name)

    cases = []

    def add(cid, name, input_type, fixture_rel, origin, bug_class, expected,
            pre, current, comparability, format_status, coverage, notes=""):
        path = FIX / fixture_rel
        data = path.read_bytes()
        cases.append({
            "id": cid,
            "name": name,
            "input_type": input_type,
            "fixture": f"fixtures/{fixture_rel}",
            "origin": origin,
            "bug_class": bug_class,
            "expected_security_property": expected,
            "pre_hardening_behavior": pre,
            "current_expected": current,
            "semantic_comparability": comparability,
            "format_status": format_status,
            "coverage": coverage,
            "sha256": sha256(data),
            "size_bytes": len(data),
            "notes": notes,
        })

    g = lambda name: f"gguf/{name}"
    t = lambda name: f"tokenizer/{name}"

    add("gguf-001", "valid-f32", "GGUF", g("valid_f32"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors"], "single f32 tensor [2,4]")
    add("gguf-002", "valid-f16", "GGUF", g("valid_f16"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors"])
    add("gguf-003", "valid-bf16", "GGUF", g("valid_bf16"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors"])
    add("gguf-004", "valid-q8_0", "GGUF", g("valid_q8_0"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors", "quantization layout"])
    add("gguf-005", "valid-q4_k", "GGUF", g("valid_q4_k"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors", "quantization layout"])
    add("gguf-006", "valid-q6_k", "GGUF", g("valid_q6_k"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors", "quantization layout"])
    add("gguf-007", "valid-metadata", "GGUF", g("valid_metadata"), "control", "control",
        "loads cleanly", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["metadata", "strings/arrays"])
    add("gguf-008", "valid-eof-exact", "GGUF", g("valid_eof_exact"), "control", "control",
        "tensor data ending exactly at EOF loads", "ACCEPT", "ACCEPT",
        "FULLY_COMPARABLE", "valid", ["extent arithmetic"])
    add("gguf-009", "valid-two-tensors", "GGUF", g("valid_two_tensors"), "control", "control",
        "disjoint tensors load", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["tensor descriptors", "overlap/range"])
    add("gguf-010", "bad-magic", "GGUF", g("bad_magic"), "regression fixture", "A",
        "structured rejection", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["header/count"])
    add("gguf-011", "unsupported-version", "GGUF", g("bad_version"), "regression fixture", "A",
        "structured rejection", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["header/count"])
    add("gguf-012", "huge-tensor-count", "GGUF", g("huge_tensor_count"), "fuzz corpus seed", "G",
        "count rejected before allocation", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["header/count"])
    add("gguf-013", "huge-kv-count", "GGUF", g("huge_kv_count"), "fuzz corpus seed", "G",
        "count rejected before allocation", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["header/count", "metadata"])
    add("gguf-014", "truncated-header", "GGUF", g("truncated_header"), "fuzz corpus seed", "A",
        "structured rejection", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["header/count"])
    add("gguf-015", "truncated-tensor-data", "GGUF", g("truncated_tensor_data"), "fuzz corpus seed", "B",
        "range past EOF rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["extent arithmetic"])
    add("gguf-016", "offset-past-eof", "GGUF", g("offset_past_eof"), "fuzz corpus seed", "B",
        "range past EOF rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["extent arithmetic"])
    add("gguf-017", "offset-u64-max", "GGUF", g("offset_overflow"), "fuzz corpus seed", "B",
        "offset overflow rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["extent arithmetic"])
    add("gguf-018", "dim-product-overflow", "GGUF", g("dim_product_overflow"), "fuzz corpus seed", "B",
        "product overflow rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["extent arithmetic", "tensor descriptors"])
    add("gguf-019", "rank-zero", "GGUF", g("rank_zero"), "fuzz corpus seed", "D",
        "invalid rank rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["tensor descriptors"])
    add("gguf-020", "rank-five", "GGUF", g("rank_five"), "fuzz corpus seed", "D",
        "invalid rank rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["tensor descriptors"])
    add("gguf-021", "zero-dimension", "GGUF", g("zero_dim"), "fuzz corpus seed", "D",
        "zero dim rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["tensor descriptors"])
    add("gguf-022", "unsupported-dtype-99", "GGUF", g("unsupported_dtype"), "fuzz corpus seed", "J",
        "clean rejection", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["tensor descriptors"])
    add("gguf-023", "q4_0-unimplemented", "GGUF", g("q4_0_unimplemented"), "fuzz corpus seed", "J",
        "clean rejection (Ember does not support q4_0)", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE", "Ember-unsupported",
        ["tensor descriptors", "quantization layout"],
        "llama.cpp supports q4_0; not a defect in either runtime")
    add("gguf-024", "q8_0-dim-misaligned", "GGUF", g("q8_0_dim_misaligned"), "fuzz corpus seed", "D",
        "contiguous dim must be 32-aligned", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["quantization layout"],
        "full-payload fixture; baseline rejects later at the compressed constructor, "
        "current rejects at the validation gate (same outcome, earlier stage)")
    add("gguf-025", "q4_k-dim-misaligned", "GGUF", g("q4_k_dim_misaligned"), "regression fixture", "D",
        "contiguous dim must be 256-aligned (llama.cpp ne[0] % QK_K == 0)",
        "SEMANTIC_MISINTERPRETATION", "STRUCTURED_REJECT", "FULLY_COMPARABLE",
        "semantically hostile", ["quantization layout"],
        "full-payload fixture; BASELINE accepts on the eager K path and dequantizes with a "
        "non-compliant block order; headline layout-gap finding")
    add("gguf-026", "q4_k-truncated", "GGUF", g("q4_k_truncated"), "fuzz corpus seed", "B",
        "range past EOF rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["extent arithmetic", "quantization layout"])
    add("gguf-027", "overlapping-ranges", "GGUF", g("overlap"), "fuzz corpus seed", "B",
        "overlap rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["overlap/range"])
    add("gguf-028", "duplicate-tensor-name", "GGUF", g("duplicate_name"), "fuzz corpus seed", "A",
        "duplicate rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["tensor descriptors"])
    add("gguf-029", "huge-string-length", "GGUF", g("huge_string_len"), "fuzz corpus seed", "G",
        "string length rejected before allocation", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "FULLY_COMPARABLE", "semantically hostile", ["strings/arrays"])
    add("gguf-030", "string-longer-than-file", "GGUF", g("string_longer_than_file"), "fuzz corpus seed", "A",
        "structured rejection", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["strings/arrays"])
    add("gguf-031", "huge-array-count", "GGUF", g("huge_array_count"), "fuzz corpus seed", "G",
        "array count rejected before allocation", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "FULLY_COMPARABLE", "semantically hostile", ["strings/arrays"])
    add("gguf-032", "deep-nested-arrays", "GGUF", g("deep_nested_arrays"), "fuzz corpus seed", "A",
        "depth cap enforced", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["metadata", "strings/arrays"])
    add("gguf-033", "bad-bool-metadata", "GGUF", g("bad_bool"), "fuzz corpus seed", "A",
        "invalid bool rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["metadata"])
    add("gguf-034", "bad-metadata-value-type", "GGUF", g("bad_value_type"), "fuzz corpus seed", "A",
        "unknown value type rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["metadata"])
    add("gguf-035", "empty-metadata-key", "GGUF", g("empty_key"), "fuzz corpus seed", "A",
        "empty key rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["metadata"])
    add("gguf-036", "bad-alignment", "GGUF", g("bad_alignment"), "fuzz corpus seed", "A",
        "non-power-of-two alignment rejected", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "FULLY_COMPARABLE", "semantically hostile",
        ["metadata", "extent arithmetic"])
    add("gguf-037", "empty-tensor-name", "GGUF", g("empty_tensor_name"), "fuzz corpus seed", "A",
        "empty name rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "format-invalid", ["tensor descriptors"])
    add("gguf-038", "offset-plus-size-overflow", "GGUF", g("range_overflow.bin"),
        "structured synthetic boundary case", "B",
        "start+byte_len wrap rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["extent arithmetic"])
    add("gguf-039", "huge-dimension-past-eof", "GGUF", g("huge_dim_past_eof.bin"),
        "structured synthetic boundary case", "B",
        "huge extent rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["extent arithmetic"])
    add("gguf-040", "tensor-count-above-abs-cap", "GGUF", g("count_above_abs_cap.bin"),
        "structured synthetic boundary case", "G",
        "count above named cap rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "FULLY_COMPARABLE", "semantically hostile", ["header/count"],
        "baseline's file-relative bound also fires on this tiny file; the absolute cap is the EmberSEC addition")
    add("gguf-041", "tiny-llama-valid", "GGUF", g("tiny_llama_valid.bin"),
        "regression fixture", "control",
        "loads and builds a model", "ACCEPT", "ACCEPT", "FULLY_COMPARABLE",
        "valid", ["model construction", "architecture metadata"])
    add("gguf-042", "llama-context-u32-max", "GGUF", g("tiny_llama_hostile_ctx.bin"),
        "fuzz corpus seed", "G",
        "config cap rejects before rope allocation",
        "PROCESS_CRASH", "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE",
        "semantically hostile", ["architecture metadata", "model construction"],
        "BASELINE: multi-TB vec! zero-fill thrashes; killed by harness timeout")
    add("gguf-043", "llama-block-count-1m", "GGUF", g("tiny_llama_hostile_layers.bin"),
        "fuzz corpus seed", "G",
        "layer cap rejects before block vector sizing",
        "STRUCTURED_REJECT", "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE",
        "semantically hostile", ["architecture metadata", "model construction"],
        "baseline rejects at the first missing per-layer tensor but only after sizing the block vector")
    add("gguf-044", "llama-missing-tensors", "GGUF", g("tiny_llama_missing_tensors.bin"),
        "fuzz corpus seed", "C",
        "inventory gate reports all missing tensors",
        "STRUCTURED_REJECT", "STRUCTURED_REJECT", "EMBER_SPECIFIC",
        "semantically hostile", ["model construction"],
        "baseline errors on the first missing tensor mid-construction; current lists all before allocating")
    add("gguf-045", "llama-odd-key-length", "GGUF", g("tiny_llama_odd_keylen.bin"),
        "fuzz-discovered class; canonical synthetic fixture", "E",
        "config rejects odd head_dim; no panic",
        "PANIC", "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE",
        "semantically hostile", ["architecture metadata", "model construction"],
        "fuzz-minimized artifact (key_length=1) panicked compute_rope_freqs in baseline")
    add("gguf-046", "llama-1d-attn-q", "GGUF", g("tiny_llama_1d_attn_q.bin"),
        "structured synthetic boundary case", "E",
        "1-D linear weight rejected; no transpose panic",
        "PANIC", "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE",
        "semantically hostile", ["tensor descriptors", "model construction"])
    add("gguf-047", "llama-rope-product-cap", "GGUF", g("tiny_llama_rope_product.bin"),
        "structured synthetic boundary case", "G",
        "rope table element product cap rejects",
        "PROCESS_CRASH", "STRUCTURED_REJECT", "PARTIALLY_COMPARABLE",
        "semantically hostile", ["architecture metadata", "model construction"],
        "individually legal caps whose product would drive a ~256 GiB rope table")
    add("gguf-048", "llama-unknown-architecture", "GGUF", g("tiny_llama_arch_gpt5.bin"),
        "structured synthetic boundary case", "C",
        "unknown architecture rejected", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "PARTIALLY_COMPARABLE", "semantically hostile", ["architecture metadata"])
    add("gguf-049", "llama-vocab-5m", "GGUF", g("tiny_llama_hostile_vocab.bin"),
        "structured synthetic boundary case", "C",
        "vocab cap rejects", "STRUCTURED_REJECT", "STRUCTURED_REJECT",
        "PARTIALLY_COMPARABLE", "semantically hostile", ["architecture metadata"])

    # ---- complete valid-model controls (runnable on Ember AND llama.cpp) --
    # These exceed the 4 KiB preference: a llama.cpp-runnable control needs
    # a real vocab (264 tokens incl. the gpt2 byte encoder) and full
    # hparams/tokenizer metadata.
    # alignment = 0: the candle divide-by-zero panic input (S4)
    (GGUF / "alignment_zero.bin").write_bytes(
        u32(MAGIC) + u32(3) + u64(0) + u64(1)
        + s(b"general.alignment") + u32(4) + u32(0))
    add("gguf-053", "alignment-zero", "GGUF", g("alignment_zero.bin"),
        "fuzz-discovered class; canonical synthetic fixture", "A",
        "alignment must be a non-zero power of two",
        "STRUCTURED_REJECT", "STRUCTURED_REJECT", "FULLY_COMPARABLE",
        "semantically hostile", ["metadata", "extent arithmetic"],
        "same bytes as the candle divide-by-zero panic input (S4): candle "
        "panics (div_ceil(0)), Ember and llama.cpp reject")

    add("gguf-050", "tiny-llama-full-valid", "GGUF", g("tiny_llama_full.bin"),
        "structured synthetic boundary case", "control",
        "loads and builds on Ember; loads and generates on llama.cpp",
        "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["model construction", "architecture metadata", "tokenizer JSON"],
        "complete 1-layer llama with 264-token gpt2-style BPE vocab; llama.cpp "
        "loads it (requires byte tokens, merges key, 32-byte tensor alignment)")
    add("gguf-051", "tiny-qwen3-full-valid", "GGUF", g("tiny_qwen3_full.bin"),
        "structured synthetic boundary case", "control",
        "loads and builds on Ember; loads and generates on llama.cpp",
        "ACCEPT", "ACCEPT", "FULLY_COMPARABLE", "valid",
        ["model construction", "architecture metadata", "tokenizer JSON"],
        "complete 1-layer qwen3 (qk norms included) with the same vocab")
    add("gguf-052", "tiny-gemma4-full-valid", "GGUF", g("tiny_gemma4_full.bin"),
        "structured synthetic boundary case", "control",
        "loads and builds on Ember; llama.cpp b5999 lacks the gemma4 arch",
        "ACCEPT", "ACCEPT", "PARTIALLY_COMPARABLE", "valid",
        ["model construction", "architecture metadata"],
        "complete 1-layer gemma4 for Ember; llama.cpp b5999 rejects with "
        "'unknown model architecture: gemma4' (arch-support gap, not a defect)")

    add("tok-001", "valid-mini-bpe", "TOKENIZER_JSON", t("valid_mini_bpe.json"),
        "control", "control", "parses and encodes", "ACCEPT", "ACCEPT",
        "TOKENIZER_ONLY", "valid", ["tokenizer JSON"])
    add("tok-002", "decoder-invalid-utf8-26b", "TOKENIZER_JSON",
        t("fuzz_decoder_invalid_utf8_26b.bin"),
        "fuzz-discovered minimized artifact (reconstructed)", "F",
        "malformed tokenizer JSON rejected; no panic",
        "PANIC", "STRUCTURED_REJECT", "TOKENIZER_ONLY", "semantically hostile",
        ["tokenizer JSON"],
        "exact 26-byte minimized reproducer of tokenizers-0.20.4 decoders/mod.rs:90 .expect(\"Helper\")")
    add("tok-003", "decoder-bad-value-15b", "TOKENIZER_JSON",
        t("fuzz_decoder_bad_value_15b.bin"),
        "fuzz-discovered minimized artifact (reconstructed)", "F",
        "malformed tokenizer JSON rejected; no panic",
        "PANIC", "STRUCTURED_REJECT", "TOKENIZER_ONLY", "semantically hostile",
        ["tokenizer JSON"], "same upstream panic site, second minimized input")
    add("tok-004", "decoder-invalid-utf8-nested", "TOKENIZER_JSON",
        t("synth_decoder_invalid_utf8_nested.bin"),
        "fuzz-discovered class; canonical synthetic fixture", "F",
        "malformed tokenizer JSON rejected; no panic",
        "PANIC", "STRUCTURED_REJECT", "TOKENIZER_ONLY", "semantically hostile",
        ["tokenizer JSON"], "same class as a 163-byte fuzz artifact whose bytes were not retained")
    add("tok-005", "truncated-json", "TOKENIZER_JSON", t("truncated.json"),
        "fuzz corpus seed", "A", "structured rejection", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "TOKENIZER_ONLY", "format-invalid", ["tokenizer JSON"])
    add("tok-006", "not-json", "TOKENIZER_JSON", t("not_json.bin"),
        "fuzz corpus seed", "A", "structured rejection", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "TOKENIZER_ONLY", "format-invalid", ["tokenizer JSON"])
    add("tok-007", "deep-nesting", "TOKENIZER_JSON", t("deep_nesting.json"),
        "fuzz corpus seed", "A", "structured rejection", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "TOKENIZER_ONLY", "semantically hostile", ["tokenizer JSON"])
    add("tok-008", "bad-utf8-vocab", "TOKENIZER_JSON", t("bad_utf8.json"),
        "fuzz corpus seed", "A", "structured rejection", "STRUCTURED_REJECT",
        "STRUCTURED_REJECT", "TOKENIZER_ONLY", "format-invalid", ["tokenizer JSON"],
        "invalid UTF-8 in a non-decoder field errors in baseline too (serde path, no panic)")
    add("tok-009", "valid-json-unknown-top-level-key", "TOKENIZER_JSON", t("huge_declared.json"),
        "fuzz corpus seed", "J",
        "valid JSON, unsupported by the tokenizers crate (unknown top-level key)",
        "STRUCTURED_REJECT", "STRUCTURED_REJECT", "TOKENIZER_ONLY",
        "Ember-unsupported", ["tokenizer JSON"],
        "the tokenizers crate rejects unknown top-level keys with a syntax-style error; "
        "not a defect in either build")

    corpus = {
        "schema_version": 1,
        "description": "EmberSEC comparative hostile-input corpus (GGUF + tokenizer JSON)",
        "generated_by": "research/embersec/comparative/gen_corpus.py",
        "case_count": len(cases),
        "cases": cases,
    }
    CORPUS.write_text(json.dumps(corpus, indent=2) + "\n")
    n = len(cases)
    hostile = sum(1 for c in cases if c["bug_class"] != "control")
    print(f"wrote {n} cases ({hostile} hostile, {n - hostile} control)")


if __name__ == "__main__":
    main()
