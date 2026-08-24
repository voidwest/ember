#!/usr/bin/env python3
"""Phase 5 Session 2 multimodal benchmark harness (Track I).

Runs one or more benchmark groups through the ember CLI and appends one JSONL
record per run with full reproduction metadata (commit, CPU, thermals, model,
workload, stage timings).

Usage: python scripts/bench_phase5.py --group audio --out results.jsonl
"""
import argparse, json, os, subprocess, sys, time

ROOT = "/home/west/ember"
E = f"{ROOT}/target/release/ember"
ENV = {
    "TEXT": "/home/west/ember/Llama-3.2-1B-Instruct-Q8_0.gguf",
    "AUDIO": "/home/west/ember-work/ultravox/audio-f32.gguf",
    "TOK": "/home/west/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B/snapshots/9535bd9b1d1dea6acafbdc4813b728796aeb28da/tokenizer.json",
    "TTS": "/home/west/ember-work/tts/outetts-gguf/OuteTTS-0.2-500M-Q8_0.gguf",
    "TTSTOK": "/home/west/ember-work/tts/outetts-hf/tokenizer.json",
    "CODEC": "/home/west/ember-work/tts/wavtokenizer-decoder-f32.gguf",
    "VITS": "/home/west/ember-work/mms-tts/ara.vits.gguf",
}
BANK = f"{ROOT}/research/banks/arabic_speech_001"


def host_meta():
    meta = {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        meta["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        meta["git_dirty"] = bool(dirty)
    except Exception:
        pass
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                meta["cpu"] = line.split(":", 1)[1].strip()
                break
        temps = [int(open(p).read()) / 1000 for p in __import__("glob").glob("/sys/class/thermal/thermal_zone*/temp")]
        if temps:
            meta["cpu_temp_c"] = max(temps)
        freqs = []
        import glob as g
        for line in g.glob("/proc/cpuinfo"):
            pass
        freqs = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if "MHz" in l]
        if freqs:
            meta["cpu_freq_mhz_avg"] = round(sum(freqs) / len(freqs), 1)
        meta["threads"] = os.cpu_count()
        la = os.getloadavg()
        meta["loadavg_1m"] = round(la[0], 2)
    except Exception:
        pass
    return meta


def run(cmd):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    return p.stdout + "\n" + p.stderr, time.time() - t0, p.returncode


def parse_ms(line, key):
    for part in line.split("|"):
        part = part.strip()
        if key in part:
            try:
                return float(part.split(key)[1].split("ms")[0].strip())
            except Exception:
                pass
    return None


def bench_tts_static(engine):
    if engine == "oute":
        cmd = ["timeout", "300", E, "tts", "--codec", ENV["CODEC"], "--model", ENV["TTS"],
               "--tokenizer", ENV["TTSTOK"], "--text", "Hello! How are you today?",
               "--out", "/tmp/opencode/p5s2/bench_oute.wav", "--max-tokens", "208"]
    else:
        cmd = ["timeout", "300", E, "tts", "--codec", ENV["CODEC"], "--vits-model", ENV["VITS"],
               "--text", "مرحبا، كيف حالك اليوم؟", "--out", "/tmp/opencode/p5s2/bench_vits.wav"]
    out, wall, rc = run(cmd)
    rec = {"workload": f"tts-static-{engine}", "returncode": rc, "wall_s": round(wall, 2)}
    for line in out.splitlines():
        if "RTF" in line and "|" in line:
            for k, tag in [("prompt_ms", "prompt"), ("prefill", "prefill"), ("gen ", "generate"), ("codec", "codec")]:
                pass
        if engine == "vits" and line.startswith("vits:"):
            rec["raw"] = line.strip()
        elif engine == "oute" and line.startswith("timings:"):
            rec["raw"] = line.strip()
    # WAV duration
    wav = "/tmp/opencode/p5s2/bench_%s.wav" % engine
    if os.path.exists(wav):
        import wave
        try:
            w = wave.open(wav)
            rec["audio_seconds"] = round(w.getnframes() / w.getframerate(), 3)
            rec["rtf"] = round(wall / max(rec["audio_seconds"], 0.01), 2)
        except Exception:
            pass
    return rec


def bench_audio_stream_validate(wav):
    name = os.path.basename(wav).replace(".wav", "")
    cmd = ["timeout", "600", E, "audio", "--model", ENV["TEXT"], "--audio-model", ENV["AUDIO"],
           "--tokenizer", ENV["TOK"], "--audio", wav, "--stream-validate"]
    out, wall, rc = run(cmd)
    rec = {"workload": f"ultravox-stream-validate-{name}", "returncode": rc, "wall_s": round(wall, 2),
           "bit_exact_all_patterns": "ALL PATTERNS BIT-EXACT" in out}
    for line in out.splitlines():
        if "finish" in line and "single push" in line:
            rec["static_finish_ms"] = parse_ms(line, "finish ")
            rec["encoder_ms"] = parse_ms(line, "enc ")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=["tts", "audio"])
    ap.add_argument("--out", default="/tmp/opencode/p5s2/bench_results.jsonl")
    args = ap.parse_args()
    meta = host_meta()
    records = []
    if args.group == "tts":
        records.append({**meta, **bench_tts_static("oute")})
        records.append({**meta, **bench_tts_static("vits")})
    elif args.group == "audio":
        bank = sorted(os.listdir(BANK))
        for wav in [os.path.join(BANK, w) for w in bank[:6] if w.endswith(".wav")]:
            records.append({**meta, **bench_audio_stream_validate(wav)})
    with open(args.out, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(json.dumps(r, ensure_ascii=False)[:200])


if __name__ == "__main__":
    main()
