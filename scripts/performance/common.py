"""Shared helpers for the Ember performance-baseline drivers."""
import json, os, subprocess, sys, time, hashlib
from pathlib import Path

REPO = Path("/home/west/ember")
BIN = REPO / "target" / "release" / "ember"
PROMPT = "في الجملة التالية، الكلمة المميزة هي: كِتَاب. اشرح معناها."
TOKENIZER = REPO / "tokenizer.json"

MODELS = {
    "llama-1b-q8": dict(path=str(REPO / "Llama-3.2-1B-Instruct-Q8_0.gguf"), arch="llama"),
    "llama-1b-q4km": dict(path=str(REPO / "Llama-3.2-1B-Instruct.Q4_K_M.gguf"), arch="llama"),
    "llama-1b-q6k": dict(path=str(REPO / "Llama-3.2-1B-Instruct.Q6_K.gguf"), arch="llama"),
    "qwen-1.5b-q8": dict(path=str(REPO / "qwen2.5-1.5b-instruct-q8_0.gguf"), arch="qwen3"),
}

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def commit():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()

def run(cmd, out_path=None, env=None, timeout=None, capture_stderr=False):
    """Run a command; tee stdout to out_path; return (rc, stdout[, stderr])."""
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, env=e, timeout=timeout)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(p.stdout)
    if capture_stderr:
        return p.returncode, p.stdout, p.stderr
    return p.returncode, p.stdout

def bench_decode(model_key, out_path, tokens=128, warmups=2, reps=5, threads=8,
                 execution="reference", profile=False, allocations=False, token_id=1):
    m = MODELS[model_key]
    cmd = [str(BIN), "bench-decode", "--model", m["path"], "--arch", m["arch"],
           "--tokens", str(tokens), "--warmups", str(warmups), "--repetitions", str(reps),
           "--token-id", str(token_id), "--execution", execution]
    if profile:
        cmd.append("--profile-operators")
    if allocations:
        cmd.append("--allocations")
    env = {"RAYON_NUM_THREADS": str(threads)}
    rc, out = run(cmd, out_path=out_path, env=env)
    if rc != 0:
        print(f"  !! bench-decode FAILED ({rc}): {out[-500:]}")
        return None
    try:
        return json.loads(out)
    except Exception as ex:
        print(f"  !! bench-decode JSON parse failed: {ex}\n{out[-500:]}")
        return None

def generate(model_key, out_dir, n_tokens=64, threads=8, capture=None, zero_layer=None,
             trace_out=None, temperature=0, prompt=PROMPT):
    m = MODELS[model_key]
    cmd = [str(BIN), "--model", m["path"], "--arch", m["arch"], "--tokenizer", str(TOKENIZER),
           "-p", prompt, "-n", str(n_tokens), "--temperature", str(temperature), "--benchmark"]
    if capture:
        cmd += ["--capture-activations", capture]
    if zero_layer:
        cmd += ["--zero-layer-output", zero_layer]
    if trace_out:
        cmd += ["--trace", "ops", "--trace-out", trace_out]
    env = {"RAYON_NUM_THREADS": str(threads)}
    rc, out, err = run(cmd, out_path=None, env=env, capture_stderr=True)
    return rc, out + "\n" + err
