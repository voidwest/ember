#!/usr/bin/env python3
"""Experiment 1: structured differential fuzzing across runtimes.

Mutations of corpus seeds (magic preserved) are executed against
ember-current, ember-baseline, llama.cpp, and candle in process-isolated
subprocesses. Outputs outcome distributions and saves any crashing input
for triage.

    python research/embersec/comparative/diff_fuzz.py --n 20000
"""

import argparse
import json
import os
import random
import struct
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_eval  # noqa: E402

OUT = HERE / "results" / "diff_fuzz"

MAGIC_BYTES = b"GGUF"


def field_offsets(data):
    """Locate mutable numeric fields: header counts, metadata value slots,
    tensor-info dims/dtype/offset slots. Returns list of (offset, width)."""
    offs = []
    if len(data) < 24 or data[:4] != MAGIC_BYTES:
        return offs
    (nt, nkv) = struct.unpack_from("<QQ", data, 8)
    # header counts themselves
    offs += [(8, 8), (16, 8)]
    pos = 24
    for _ in range(min(nkv, 2000)):
        if pos + 8 > len(data):
            return offs
        (klen,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        key = data[pos:pos + klen]
        pos += klen
        if pos + 4 > len(data):
            return offs
        (vtype,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if vtype in (4, 5, 6):
            offs.append((pos, 4))
            pos += 4
        elif vtype == 7:
            offs.append((pos, 1))
            pos += 1
        elif vtype == 8:
            if pos + 8 > len(data):
                return offs
            (slen,) = struct.unpack_from("<Q", data, pos)
            offs.append((pos, 8))  # string length slot
            pos += 8 + slen
        elif vtype == 9:
            if pos + 12 > len(data):
                return offs
            (et, cnt) = struct.unpack_from("<IQ", data, pos)
            offs.append((pos + 4, 8))  # array count slot
            pos += 12
            for _ in range(min(cnt, 5000)):
                if et == 8:
                    if pos + 8 > len(data):
                        return offs
                    (slen,) = struct.unpack_from("<Q", data, pos)
                    offs.append((pos, 8))
                    pos += 8 + slen
                elif et in (0, 7):
                    pos += 1
                elif et in (2, 3):
                    pos += 2
                elif et in (4, 5, 6):
                    offs.append((pos, 4))
                    pos += 4
                elif et in (10, 11, 12):
                    offs.append((pos, 8))
                    pos += 8
                else:
                    return offs
        elif vtype == 10:
            offs.append((pos, 8))
            pos += 8
        else:
            return offs
    # tensor infos
    for _ in range(min(nt, 2000)):
        if pos + 8 > len(data):
            return offs
        (nlen,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + nlen
        if pos + 4 > len(data):
            return offs
        (nd,) = struct.unpack_from("<I", data, pos)
        offs.append((pos, 4))  # rank slot
        pos += 4
        for _ in range(min(nd, 8)):
            offs.append((pos, 8))  # dim slot
            pos += 8
        offs.append((pos, 4))  # dtype slot
        pos += 4
        offs.append((pos, 8))  # offset slot
        pos += 8
    return offs


BOUNDARY64 = [0, 1, 2, 31, 32, 33, 255, 256, 257, 2**31 - 1, 2**31, 2**32 - 1,
              2**32, 2**40, 2**63 - 1, 2**63, 2**64 - 1]
BOUNDARY32 = [0, 1, 2, 7, 8, 9, 15, 30, 31, 32, 33, 99, 255, 2**31 - 1, 2**31,
              2**32 - 1]


CONFIG_BOUNDARY64 = [0, 1, 2, 3, 4, 5, 7, 31, 255, 4096, 1 << 20, 1 << 24,
                     2**31 - 1, 2**31, 2**32 - 1, 2**32, 2**63 - 1]


def metadata_value_slots(data):
    """Offsets of scalar metadata VALUE slots only (u32/f32), excluding
    string lengths, array counts, and everything in the tensor-info
    section. Patching only these keeps the file loadable, so mutations
    reach the model-construction layer."""
    slots = []
    if len(data) < 24 or data[:4] != MAGIC_BYTES:
        return slots
    (nt, nkv) = struct.unpack_from("<QQ", data, 8)
    pos = 24
    for _ in range(min(nkv, 2000)):
        if pos + 8 > len(data):
            return slots
        (klen,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        key = data[pos:pos + klen]
        pos += klen
        if pos + 4 > len(data):
            return slots
        (vtype,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if vtype in (4, 6):
            slots.append(pos)  # u32 / f32 scalar value slot
            pos += 4
        elif vtype == 5:
            pos += 4
        elif vtype == 7:
            pos += 1
        elif vtype == 8:
            if pos + 8 > len(data):
                return slots
            (slen,) = struct.unpack_from("<Q", data, pos)
            pos += 8 + slen
        elif vtype == 9:
            if pos + 12 > len(data):
                return slots
            (et, cnt) = struct.unpack_from("<IQ", data, pos)
            pos += 12
            for _ in range(min(cnt, 5000)):
                if et == 8:
                    if pos + 8 > len(data):
                        return slots
                    (slen,) = struct.unpack_from("<Q", data, pos)
                    pos += 8 + slen
                elif et in (0, 7):
                    pos += 1
                elif et in (2, 3):
                    pos += 2
                elif et in (4, 5, 6):
                    pos += 4
                elif et in (10, 11, 12):
                    pos += 8
                else:
                    return slots
        elif vtype == 10:
            pos += 8
        else:
            return slots
    return slots


def mutate_construction(seed, rng):
    """Construction-layer mutation: patch 1-3 scalar metadata values of a
    loadable model to boundary values (odd head dims, huge context,
    absurd counts, ...). The file stays loadable; the model-construction
    layer sees the hostile config."""
    data = bytearray(seed)
    slots = metadata_value_slots(bytes(data))
    if not slots or len(data) < 5:
        return bytes(data)
    for _ in range(rng.randint(1, 3)):
        off = rng.choice(slots)
        if off + 4 > len(data):
            continue
        struct.pack_into("<I", data, off, rng.choice(CONFIG_BOUNDARY64) & 0xFFFFFFFF)
    if rng.random() < 0.15:
        # a raw tweak inside the metadata region (before the first tensor info)
        first_tensor = len(data)
        (nt, nkv) = struct.unpack_from("<QQ", data, 8)
        # approximate: keep tweaks early in the file
        p = rng.randrange(24, min(len(data), 4096))
        data[p] ^= rng.randrange(1, 256)
    return bytes(data)


def mutate(seed, rng):
    data = bytearray(seed)
    if len(data) < 5:
        return bytes(data)
    slots = field_offsets(bytes(data))
    if rng.random() < 0.45 and slots:
        # structured: patch one numeric field to a boundary value
        off, width = rng.choice(slots)
        if off + width <= len(data):
            if width == 8:
                struct.pack_into("<Q", data, off, rng.choice(BOUNDARY64))
            elif width == 4:
                struct.pack_into("<I", data, off, rng.choice(BOUNDARY32))
            else:
                data[off] = rng.choice([0, 1, 2, 0x7F, 0x80, 0xFF])
        # maybe also a raw tweak
        if rng.random() < 0.3:
            p = rng.randrange(4, len(data))
            data[p] ^= rng.randrange(1, 256)
    else:
        # raw: 1..6 byte edits after the magic
        for _ in range(rng.randint(1, 6)):
            if len(data) <= 5:
                break
            p = rng.randrange(4, len(data))
            op = rng.random()
            if op < 0.5:
                data[p] ^= rng.randrange(1, 256)
            elif op < 0.75:
                data[p] = rng.choice([0x00, 0xFF, 0x7F, 0x80, 0x01])
            elif op < 0.9:
                data.insert(p, rng.randrange(256))
            else:
                del data[p]
    if rng.random() < 0.2 and len(data) > 5:
        data = data[: rng.randrange(5, len(data) + 1)]
    # keep the magic (if the seed had it) so mutations reach deep paths
    if seed[:4] == MAGIC_BYTES and (len(data) < 4 or data[:4] != MAGIC_BYTES):
        data[:4] = MAGIC_BYTES
    return bytes(data)


_CONSTRUCTION_MODE = {"value": False}


def set_construction_mode(on):
    _CONSTRUCTION_MODE["value"] = on


def build_target_cmds(target_cfg, blob_path):
    if target_cfg["kind"] == "ember":
        stage = "gguf_model_check" if _CONSTRUCTION_MODE["value"] else "gguf_load_check"
        return [(run_eval.resolve_harness_binary(target_cfg),
                 stage, "--exact", "--nocapture"),
                {"EMBERSEC_FIXTURE": str(blob_path)}]
    if target_cfg["kind"] == "llama-cpp":
        cmd = [target_cfg["binary"]] + [a.replace("{fixture}", str(blob_path))
                                        for a in target_cfg.get("args", [])]
        return cmd, {}
    if target_cfg["kind"] == "llama-cpp-loader":
        return [target_cfg["binary"], str(blob_path)], {}
    if target_cfg["kind"] == "candle":
        return [target_cfg["binary"], str(blob_path)], {}
    raise ValueError(target_cfg["kind"])


def run_case(args):
    target_cfg, blob_path, timeout = args
    with open(blob_path, "rb") as f:
        content = f.read()
    cmd, extra_env = build_target_cmds(target_cfg, blob_path)
    env = dict(os.environ)
    env.update(extra_env)
    res = run_eval.run_with_rusage(cmd, env, timeout)
    kind = target_cfg["kind"]
    if kind == "ember":
        outcome = run_eval.classify_ember(res)
    elif kind == "llama-cpp":
        outcome = run_eval.classify_llamacpp(res)
    elif kind == "llama-cpp-loader":
        outcome = run_eval.classify_llama_loader(res)
    else:
        outcome = run_eval.classify_candle(res)
    return outcome, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--targets", default="ember-current,ember-baseline,llama-cpp,candle")
    ap.add_argument("--mode", choices=["raw", "construction"], default="raw",
                    help="raw: all seeds + field slots; construction: metadata-only "
                         "patches on the valid models (reaches model construction)")
    args = ap.parse_args()

    envs = json.loads((HERE / "environments.json").read_text())
    targets = {t: envs["targets"][t] for t in args.targets.split(",")}
    corpus = json.loads((HERE / "corpus.json").read_text())

    seeds = []
    for case in corpus["cases"]:
        if case["input_type"] == "TOKENIZER_JSON":
            continue
        if args.mode == "construction" and not (
            case["id"] in ("gguf-050", "gguf-051", "gguf-052")
        ):
            continue
        seeds.append((HERE / case["fixture"]).read_bytes())
    # keep seeds small for speed; the 264-token models are the big ones
    set_construction_mode(args.mode == "construction")
    rng = random.Random(args.seed)
    mut = mutate_construction if args.mode == "construction" else mutate
    mutations = []
    for i in range(args.n):
        seed = rng.choice(seeds)
        mutations.append(mut(seed, rng))

    blob_dir = OUT / "blobs"
    blob_dir.mkdir(parents=True, exist_ok=True)
    crash_dir = OUT / "crashes"
    crash_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    per_target = {t: Counter() for t in targets}
    crash_count = 0
    saved = set()
    # Sequential execution: the baseline TIMEOUT cases (multi-TB zero-fill
    # under a short timeout) create heavy memory pressure; concurrent
    # workers caused OOM-killed workers and corrupted pool results on this
    # host. Every case result is appended to a JSONL log so a partial run
    # is never lost and results are reproducible.
    run_tag = f"{args.mode}-{args.n}-{args.seed}"
    log_path = OUT / f"log_{run_tag}.jsonl"
    # A run tag identifies one deterministic campaign.  Truncate any prior
    # log before appending records so rerunning the same tag cannot silently
    # duplicate (or mix versions of) the campaign in its audit trail.
    log_path.write_text("")
    for i, blob in enumerate(mutations):
        blob_path = blob_dir / f"m{i:05d}.bin"
        blob_path.write_bytes(blob)
        for tname, cfg in targets.items():
            outcome, res = run_case((cfg, str(blob_path), args.timeout))
            per_target[tname][outcome] += 1
            record = {"i": i, "target": tname, "outcome": outcome,
                      "exit_code": res["exit_code"], "blob": blob_path.name}
            with open(log_path, "a") as lf:
                lf.write(json.dumps(record) + "\n")
            if outcome in ("PANIC", "PROCESS_CRASH", "TIMEOUT", "RESOURCE_LIMIT"):
                # save the input + stderr for triage (dedup by content hash)
                import hashlib
                h = hashlib.sha256(blob).hexdigest()[:16]
                key = (tname, h)
                if key not in saved:
                    saved.add(key)
                    crash_count += 1
                    d = crash_dir / tname
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{h}.bin").write_bytes(blob)
                    (d / f"{h}.stderr").write_text(res["stderr"][-800:])
        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t0
            print(f"  {i+1}/{args.n} cases, {elapsed:.0f}s, "
                  f"failures so far: {crash_count}", flush=True)

    summary = {
        "mode": args.mode,
        "n_mutations": args.n,
        "seed_cases": len(seeds),
        "rng_seed": args.seed,
        "timeout_s": args.timeout,
        "per_target": {t: dict(c) for t, c in per_target.items()},
        "failure_inputs_saved": crash_count,
    }
    summary_json = json.dumps(summary, indent=2) + "\n"
    # Keep a run-specific summary alongside the legacy latest-run filename;
    # otherwise running raw and construction campaigns overwrites evidence
    # needed to reproduce the other campaign.
    (OUT / f"summary_{run_tag}.json").write_text(summary_json)
    (OUT / "summary.json").write_text(summary_json)
    for t, c in per_target.items():
        print(f"{t:18s}", dict(c))


if __name__ == "__main__":
    main()
