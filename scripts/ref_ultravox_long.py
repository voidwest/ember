#!/usr/bin/env python3
"""Long-form (>30 s) Ultravox reference: the chunked protocol.

Implements exactly the archived UltravoxProcessor._chunk_and_pad_audio +
ModifiedWhisperEncoder padding-mask behavior on top of the native Whisper
encoder components, then compares against ember's chunked path:

  - mel over the FULL audio (one global max-8 floor), windows of 3000 frames;
  - continuation windows zero-padded in the mel domain to 3000;
  - per-window encoder attention mask over positions >= ceil(valid/2)
    (additive torch.finfo(f32).min, like get_extended_attention_mask);
  - projected rows truncated to ceil(valid/16) per window and concatenated;
  - one <|audio|> placeholder expands to the concatenated token count.

Usage:
  python scripts/ref_ultravox_long.py <wav> "<prompt>" <out_dir> [max_new_tokens]
"""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder

sys.path.insert(0, os.path.dirname(__file__))
from ref_ultravox import (  # noqa: E402
    DATE,
    FE_ID,
    MODEL_DIR,
    TEMPLATE,
    ULTRAVOX_SAFETENSORS,
    load_wav_16k_mono,
    whisper_mel,
)

CONTEXT_FRAMES = 3000
DS = 2  # encoder_ds_factor (single stride-2 conv)
STACK = 8


def load_encoder() -> WhisperEncoder:
    sd = load_file(ULTRAVOX_SAFETENSORS)
    cfg = WhisperConfig.from_pretrained(FE_ID)
    encoder = WhisperEncoder(cfg).eval()
    remap = {}
    prefix_map = {
        "conv1.weight": "conv1.weight",
        "conv1.bias": "conv1.bias",
        "conv2.weight": "conv2.weight",
        "conv2.bias": "conv2.bias",
        "embed_positions.weight": "embed_positions.weight",
        "layer_norm.weight": "layer_norm.weight",
        "layer_norm.bias": "layer_norm.bias",
    }
    for k, v in sd.items():
        if not k.startswith("audio_tower."):
            continue
        name = k[len("audio_tower."):]
        if name in prefix_map:
            remap[f"encoder.{prefix_map[name]}"] = v.to(torch.float32)
        elif name.startswith("layers."):
            rest = name.split(".", 2)[2] if name.count(".") >= 2 else ""
            idx = name.split(".")[1]
            remap[f"encoder.layers.{idx}.{rest}"] = v.to(torch.float32)
    missing, _unexpected = encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in remap.items()}, strict=False
    )
    assert not missing, f"missing encoder params: {missing}"
    return encoder


def encode_window(encoder, mel_win: np.ndarray, valid_frames: int):
    """One window through conv + pos + layers with the padding mask."""
    x = torch.tensor(mel_win, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        hidden = torch.nn.functional.gelu(encoder.conv1(x))
        hidden = torch.nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        pos = encoder.embed_positions.weight[: hidden.size(1)]
        hidden = hidden + pos
        # additive extended attention mask over padded output positions
        t2 = hidden.size(1)
        valid_out = min((valid_frames + DS - 1) // DS, t2)
        bias = torch.zeros(t2)
        bias[valid_out:] = torch.finfo(torch.float32).min
        ext = bias.view(1, 1, 1, t2)  # broadcast over query rows and heads
        for layer in encoder.layers:
            residual = hidden
            normed = layer.self_attn_layer_norm(hidden)
            attn, _ = layer.self_attn(hidden_states=normed, attention_mask=ext)
            hidden = residual + attn
            residual = hidden
            normed = layer.final_layer_norm(hidden)
            mlp = layer.fc2(torch.nn.functional.gelu(layer.fc1(normed)))
            hidden = residual + mlp
        enc_final = encoder.layer_norm(hidden)
    return enc_final[0].numpy(), hidden.size(1)


def project(sd, enc_out: np.ndarray) -> np.ndarray:
    stack_factor = STACK
    eps = 1e-6
    ln_pre = sd["multi_modal_projector.ln_pre.weight"].to(torch.float32)
    w1 = sd["multi_modal_projector.linear_1.weight"].to(torch.float32)
    ln_mid = sd["multi_modal_projector.ln_mid.weight"].to(torch.float32)
    w2 = sd["multi_modal_projector.linear_2.weight"].to(torch.float32)

    def rms_norm(x, weight):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    x = torch.tensor(enc_out, dtype=torch.float32)
    t2 = x.shape[0]
    t_pad = (t2 + stack_factor - 1) // stack_factor * stack_factor
    stacked = torch.nn.functional.pad(x, (0, 0, 0, t_pad - t2))
    stacked = stacked.view(t_pad // stack_factor, -1)
    h = rms_norm(stacked, ln_pre) @ w1.T
    value, gate = h.chunk(2, dim=-1)
    h = torch.nn.functional.silu(gate) * value
    h = rms_norm(h, ln_mid) @ w2.T
    return h.numpy()


def main() -> None:
    wav_path, prompt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    os.makedirs(out_dir, exist_ok=True)

    samples, rate = load_wav_16k_mono(wav_path)
    assert rate == 16_000
    from transformers import WhisperFeatureExtractor

    fe = WhisperFeatureExtractor.from_pretrained(FE_ID)
    mel, n_frames = whisper_mel(samples, fe)
    np.save(os.path.join(out_dir, "2_mel_features.npy"), mel)

    # ---- chunking (reference _chunk_and_pad_audio semantics) -------------
    windows = []
    off = 0
    while True:
        valid = min(CONTEXT_FRAMES, n_frames - off)
        windows.append((off, valid))
        if off + CONTEXT_FRAMES >= n_frames:
            break
        off += CONTEXT_FRAMES
    print(f"mel frames {n_frames}; {len(windows)} windows: {windows}")

    encoder = load_encoder()
    sd = load_file(ULTRAVOX_SAFETENSORS)
    features_per_token = []
    last_enc = None
    for i, (off, valid) in enumerate(windows):
        win = mel[:, off : off + valid]
        if i > 0 and valid < CONTEXT_FRAMES:
            win = np.pad(win, ((0, 0), (0, CONTEXT_FRAMES - valid)))
        enc_out, t2 = encode_window(encoder, win, valid)
        if i == len(windows) - 1:
            last_enc = enc_out
            np.save(os.path.join(out_dir, "5_encoder_output_last_window.npy"), enc_out)
        proj = project(sd, enc_out)
        token_len = (valid + DS * STACK - 1) // (DS * STACK)
        assert token_len <= proj.shape[0]
        features_per_token.append(proj[:token_len])
        print(f"window {i}: valid={valid} t2={t2} tokens={token_len}")

    h_all = np.concatenate(features_per_token, axis=0)
    np.save(os.path.join(out_dir, "6_projector_output.npy"), h_all)
    n_audio_tokens = h_all.shape[0]

    # ---- text + merge -----------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    rendered = TEMPLATE.format(content=prompt)
    parts = rendered.split("<|audio|>")
    assert len(parts) == 2, "single-placeholder validation"
    ids = []
    start = None
    for i, part in enumerate(parts):
        ids.extend(tok(part, add_special_tokens=False)["input_ids"])
        if i == 0:
            start = len(ids)
            ids.extend([eot] * n_audio_tokens)
    input_ids = torch.tensor([ids])

    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    embeds = model.get_input_embeddings()(input_ids)
    merged = embeds.clone()
    merged[0, start : start + n_audio_tokens] = torch.tensor(h_all)
    np.save(os.path.join(out_dir, "7_assembled_embeddings.npy"), merged[0].detach().numpy())
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids[0].numpy())

    with torch.no_grad():
        out = model(inputs_embeds=merged, use_cache=True)
        past = out.past_key_values
        step_logits = [out.logits[0, -1].numpy()]
        gen_ids = [int(np.argmax(step_logits[-1]))]
        cur = torch.tensor([[gen_ids[-1]]])
        eos_ids = {tok.convert_tokens_to_ids("<|eot_id|>"), tok.eos_token_id}
        for _ in range(max_new - 1):
            if gen_ids[-1] in eos_ids:
                break
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[0, -1].numpy()
            step_logits.append(lg)
            nxt = int(np.argmax(lg))
            gen_ids.append(nxt)
            cur = torch.tensor([[nxt]])
    np.save(os.path.join(out_dir, "8_first_logits.npy"), step_logits[0])
    np.save(os.path.join(out_dir, "step_logits.npy"), np.stack(step_logits))
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    manifest = {
        "model": MODEL_DIR,
        "audio": wav_path,
        "prompt": prompt,
        "duration_s": len(samples) / rate,
        "mel_frames": n_frames,
        "n_windows": len(windows),
        "windows": windows,
        "n_audio_tokens": n_audio_tokens,
        "generated_text": gen_text,
        "gen_ids": gen_ids,
        "shapes": {
            "2_mel_features": list(mel.shape),
            "6_projector_output": list(h_all.shape),
            "7_assembled_embeddings": list(merged[0].shape),
        },
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print("gen:", gen_text)


if __name__ == "__main__":
    main()
