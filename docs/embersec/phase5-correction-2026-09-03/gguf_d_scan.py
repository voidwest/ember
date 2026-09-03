#!/usr/bin/env python3
"""EmberSEC Phase V Step 1: real per-block f16 scale distribution from GGUFs.

Standalone struct-based GGUF header walk (no torch / llama-cpp / ember dep).
For every Q8_0 / Q4_K / Q6_K tensor, seeks to tensor data and reads ONLY the
2-byte scale headers per block (chunked sequential preads; tensor payload
bytes are never retained in RAM).

  Q8_0 stride  34, d  @ off 0
  Q4_K stride 144, d  @ off 0, min @ off 2
  Q6_K stride 210, d  @ off 208

Usage:
  python3 gguf_d_scan.py <model.gguf> [...] [--out d_distribution.json
      --samples samples.json --blocks-per-chunk 524288]

Emits per (model, dtype): n_blocks, quantiles, non-finite count,
exponent-bucket fractions, log10 histogram. Prints a compact table.
"""
import json
import os
import struct
import sys

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# dtype code -> (name, elems_per_block, stride, d_offsets, extra_offsets)
TARGETS = {
    8: ("Q8_0", 32, 34, (0,), ()),
    12: ("Q4_K", 256, 144, (0,), (2,)),   # extra = min word
    14: ("Q6_K", 256, 210, (208,), ()),
}
assert set(TARGETS) == {8, 12, 14}

FIXED_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
               10: 8, 11: 8, 12: 8}  # type 8=str, 9=array handled separately


class Reader:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.size = os.fstat(self.f.fileno()).st_size

    def read(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise ValueError(f"short read: want {n}, got {len(b)}")
        return b

    def u32(self):
        return struct.unpack("<I", self.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.read(8))[0]

    def pos(self):
        return self.f.tell()

    def close(self):
        self.f.close()


def skip_value(r, vtype, alignment_box):
    """Skip one GGUF metadata value; capture general.alignment if seen."""
    if vtype in FIXED_SIZES:
        r.read(FIXED_SIZES[vtype])
    elif vtype == 8:  # string
        n = r.u64()
        r.read(n)
    elif vtype == 9:  # array
        etype = r.u32()
        n = r.u64()
        for _ in range(n):
            skip_value(r, etype, None)
    else:
        raise ValueError(f"unsupported GGUF value type {vtype}")


def read_value_for_alignment(r, vtype):
    if vtype == 4:  # u32
        return r.u32()
    if vtype == 10:  # u64
        return r.u64()
    skip_value(r, vtype, None)
    return None


def parse_header(path):
    r = Reader(path)
    try:
        magic = r.u32()
        if magic != GGUF_MAGIC:
            raise ValueError(f"{path}: bad magic {magic:#x}")
        ver = r.u32()
        if ver != GGUF_VERSION:
            raise ValueError(f"{path}: unsupported version {ver}")
        n_tensors = r.u64()
        n_kv = r.u64()
        alignment = DEFAULT_ALIGNMENT
        for _ in range(n_kv):
            klen = r.u64()
            key = r.read(klen).decode("utf-8", errors="replace")
            vtype = r.u32()
            if key == "general.alignment":
                v = read_value_for_alignment(r, vtype)
                if v is not None:
                    alignment = v
            else:
                skip_value(r, vtype, None)
        infos = []
        for _ in range(n_tensors):
            nlen = r.u64()
            name = r.read(nlen).decode("utf-8", errors="replace")
            ndims = r.u32()
            if not 1 <= ndims <= 4:
                raise ValueError(f"tensor {name}: bad ndims {ndims}")
            dims = [r.u64() for _ in range(ndims)]
            dtype = r.u32()
            offset = r.u64()
            infos.append({"name": name, "dims": dims, "dtype": dtype,
                          "offset": offset})
        if alignment == 0 or (alignment & (alignment - 1)):
            raise ValueError(f"bad alignment {alignment}")
        data_start = (r.pos() + alignment - 1) & ~(alignment - 1)
        return infos, data_start, r.size
    finally:
        r.close()


def scan_tensor(fd, base, n_blocks, stride, offs_list, blocks_per_chunk):
    """Yield uint16 header words for each offset in offs_list.

    Returns list of uint16 ndarrays (one per offset), concatenated across
    chunks. Only header bytes are parsed; payload bytes are discarded.
    """
    accs = [[] for _ in offs_list]
    remaining = n_blocks
    pos = base
    chunk_blocks = blocks_per_chunk
    while remaining > 0:
        take = min(remaining, chunk_blocks)
        nbytes = take * stride
        buf = os.pread(fd, nbytes, pos)
        if len(buf) != nbytes:
            raise ValueError(f"short pread at {pos}: want {nbytes}")
        a = np.frombuffer(buf, dtype=np.uint8).reshape(take, stride)
        for i, off in enumerate(offs_list):
            lo = a[:, off].astype(np.uint16)
            hi = a[:, off + 1].astype(np.uint16)
            accs[i].append((lo | (hi << 8)).copy())
        del buf, a
        pos += nbytes
        remaining -= take
    return [np.concatenate(x) if x else np.zeros(0, np.uint16) for x in accs]


def summarize(bits, label):
    f32 = bits.view(np.float16).astype(np.float32)
    finite = f32[np.isfinite(f32)]
    n = int(bits.size)
    n_fin = int(finite.size)
    n_nonfin = n - n_fin
    out = {"n_blocks": n, "n_nonfinite": n_nonfin}
    if n_fin:
        qs = np.percentile(finite, [0, 1, 5, 25, 50, 75, 95, 99, 100])
        out.update({"min": float(qs[0]), "p1": float(qs[1]),
                    "p5": float(qs[2]), "p25": float(qs[3]),
                    "p50": float(qs[4]), "p75": float(qs[5]),
                    "p95": float(qs[6]), "p99": float(qs[7]),
                    "max": float(qs[8])})
        pos = finite[finite > 0]
        out["n_zero"] = int(np.sum(finite == 0))
        out["n_negative"] = int(np.sum(finite < 0))
        if pos.size:
            lo = float(np.floor(np.log10(pos.min())))
            hi = float(np.ceil(np.log10(pos.max())))
            if hi <= lo:
                hi = lo + 1.0
            counts, edges = np.histogram(np.log10(pos), bins=20,
                                         range=(lo, hi))
            out["histogram_log10"] = {"lo": lo, "hi": hi,
                                      "counts": [int(c) for c in counts]}
        else:
            out["histogram_log10"] = {"lo": 0.0, "hi": 0.0,
                                      "counts": [0] * 20}
    exp = ((bits.astype(np.uint32) >> 10) & 0x1F)
    bc = np.bincount(exp, minlength=32)
    out["frac_by_exponent_bucket"] = {str(e): float(c) / n for e, c in
                                      enumerate(bc)}
    out["label"] = label
    return out


def scan_model(path, blocks_per_chunk=524288, max_samples=5):
    infos, data_start, fsize = parse_header(path)
    fd = os.open(path, os.O_RDONLY)
    per_dtype_bits = {}
    per_dtype_min = {}
    per_dtype_tensors = {}
    samples = []
    try:
        for info in infos:
            code = info["dtype"]
            if code not in TARGETS:
                continue
            name, epb, stride, d_offs, extra = TARGETS[code]
            nelem = 1
            for d in info["dims"]:
                nelem *= d
            if nelem % epb:
                print(f"  WARN {info['name']}: {nelem} elems not a "
                      f"multiple of {epb}; skipping")
                continue
            n_blocks = nelem // epb
            byte_len = n_blocks * stride
            base = data_start + info["offset"]
            if base + byte_len > fsize:
                raise ValueError(f"tensor {info['name']} range exceeds file")
            got = scan_tensor(fd, base, n_blocks, stride,
                              list(d_offs) + list(extra), blocks_per_chunk)
            per_dtype_bits.setdefault(code, []).append(got[0])
            if extra:
                per_dtype_min.setdefault(code, []).append(got[1])
            per_dtype_tensors.setdefault(code, []).append(info["name"])
            if len(samples) < 500 and code in TARGETS:
                # keep first-block sample of first few tensors per dtype
                taken = sum(1 for s in samples if s["dtype"] == code)
                if taken < max_samples:
                    samples.append({"tensor": info["name"], "dtype": code,
                                    "block": 0,
                                    "d_bits": f"0x{int(got[0][0]):04x}"})
    finally:
        os.close(fd)
    result = {}
    for code, parts in per_dtype_bits.items():
        bits = np.concatenate(parts)
        del parts
        s = summarize(bits, TARGETS[code][0])
        s["n_tensors"] = len(per_dtype_tensors[code])
        if code in per_dtype_min:
            mb = np.concatenate(per_dtype_min[code])
            mf = mb.view(np.float16).astype(np.float32)
            fin = mf[np.isfinite(mf)]
            s["min_word"] = {"n_nonfinite":
                             int(mb.size - fin.size),
                             "min": float(fin.min()) if fin.size else None,
                             "p50": float(np.median(fin)) if fin.size else None,
                             "max": float(fin.max()) if fin.size else None}
        result[str(code)] = s
        del bits
    return result, samples


def main():
    argv = sys.argv[1:]
    out = "/tmp/opencode/phase5/d_distribution.json"
    samp_out = "/tmp/opencode/phase5/d_samples.json"
    args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = argv[i + 1]; i += 2
        elif argv[i] == "--samples":
            samp_out = argv[i + 1]; i += 2
        elif argv[i].startswith("--"):
            print(f"unknown flag {argv[i]}"); sys.exit(2)
        else:
            args.append(argv[i]); i += 1
    if not args:
        print("usage: gguf_d_scan.py <model.gguf> [...]")
        sys.exit(2)
    all_res, all_samples = {}, {}
    for path in args:
        print(f"== {path} ({os.path.getsize(path)/2**30:.2f} GiB) ==",
              flush=True)
        res, samples = scan_model(path)
        all_res[os.path.basename(path)] = res
        all_samples[os.path.basename(path)] = samples
        for code, s in sorted(res.items()):
            nm = TARGETS[int(code)][0]
            print(f"  {nm}: tensors={s['n_tensors']} "
                  f"n={s['n_blocks']} nonfinite={s['n_nonfinite']} "
                  f"min={s.get('min')} p50={s.get('p50')} "
                  f"p95={s.get('p95')} max={s.get('max')}", flush=True)
    with open(out, "w") as f:
        json.dump(all_res, f, indent=1)
    with open(samp_out, "w") as f:
        json.dump(all_samples, f, indent=1)
    print(f"wrote {out} and {samp_out}")


if __name__ == "__main__":
    main()
