#!/usr/bin/env python3
"""Capture reference Ultravox-v0.5 activations from transformers components.

Validates the ember audio pipeline against the upstream reference at every
boundary:
  0. normalized waveform samples      (16 kHz mono f32)
  2. mel features                     (WhisperFeatureExtractor)
  3. conv1 output                     (encoder.conv1 + gelu)
  4. encoder layer outputs            (selected layers)
  5. final encoder output             (layer_norm)
  6. projector output                 (stack -> RMSNorm -> swiglu MLP)
  7. assembled LLM embeddings         (eot-run scatter)
  8. first LLM logits                 (prefill logits at last position)
  9. short greedy generation          (fp32)

Usage:
  python scripts/ref_ultravox.py <wav> "<prompt with <|audio|>>" <out_dir> [max_new_tokens]

The composition mirrors fixie-ai/ultravox v0_5 modeling code (archived
transformers implementation): ModifiedWhisperEncoder + SwiGLU projector +
scatter-over-eot-runs merge. Everything runs fp32 on CPU with the same
weights ember consumes (converted GGUFs come from the same safetensors).
"""
import json
import os
import sys

import numpy as np
import torch
import wave
from safetensors.torch import load_file
from transformers import AutoTokenizer
from transformers.models.whisper.modeling_whisper import WhisperConfig, WhisperEncoder
from transformers.audio_utils import spectrogram, window_function
from transformers.models.whisper.feature_extraction_whisper import (
    WhisperFeatureExtractor,
)

MODEL_DIR = "/home/west/ember-work/llama32"
ULTRAVOX_SAFETENSORS = "/home/west/ember-work/ultravox/model.safetensors"
FE_ID = "openai/whisper-large-v3-turbo"

DATE = "01 Jan 2026"
TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\nToday Date: " + DATE + "\n\n"
    "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    "{content}"
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


def load_wav_16k_mono(path: str) -> np.ndarray:
    """Decode PCM WAV to mono float32 [-1, 1] at its native rate."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        n = w.getnframes()
        raw = w.readframes(n)
    assert width == 2, "reference loader expects int16 wav fixtures"
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, rate


def whisper_mel(samples: np.ndarray, fe: WhisperFeatureExtractor):
    """Replicate _np_extract_fbank_features (f64 STFT path) + frame count."""
    window = window_function(fe.n_fft, "hann")
    log_spec = spectrogram(
        samples,
        window,
        frame_length=fe.n_fft,
        hop_length=fe.hop_length,
        power=2.0,
        dither=0.0,
        mel_filters=fe.mel_filters,
        log_mel="log10",
    )
    log_spec = log_spec[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32), log_spec.shape[1]


def main() -> None:
    wav_path, prompt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    os.makedirs(out_dir, exist_ok=True)

    # ---- 0. waveform ----------------------------------------------------
    samples, rate = load_wav_16k_mono(wav_path)
    assert rate == 16_000, "validation fixtures are 16 kHz; resampling is off the reference path"
    np.save(os.path.join(out_dir, "0_waveform.npy"), samples.astype(np.float32))

    # ---- 2. mel features -------------------------------------------------
    fe = WhisperFeatureExtractor.from_pretrained(FE_ID)
    mel, n_frames = whisper_mel(samples, fe)
    np.save(os.path.join(out_dir, "2_mel_features.npy"), mel)

    # ---- 3-5. whisper encoder -------------------------------------------
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
    missing, unexpected = encoder.load_state_dict(
        {k.removeprefix("encoder."): v for k, v in remap.items()}, strict=False
    )
    assert not missing, f"missing encoder params: {missing}"

    input_features = torch.tensor(mel, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        conv1_out = torch.nn.functional.gelu(encoder.conv1(input_features))

    np.save(os.path.join(out_dir, "3_conv1_output.npy"), conv1_out[0].numpy())
    # transformers 5.x WhisperEncoder.forward requires exactly-3000-frame
    # input (the check ultravox's ModifiedWhisperEncoder relaxes), so run
    # the stack manually on the natural-length features
    with torch.no_grad():
        hidden = conv1_out
        hidden = torch.nn.functional.gelu(encoder.conv2(hidden)).permute(0, 2, 1)
        pos = encoder.embed_positions.weight[: hidden.size(1)]
        hidden = hidden + pos
        layer_outputs = []
        for layer in encoder.layers:
            residual = hidden
            normed = layer.self_attn_layer_norm(hidden)
            attn, _ = layer.self_attn(hidden_states=normed, attention_mask=None)
            hidden = residual + attn
            residual = hidden
            normed = layer.final_layer_norm(hidden)
            mlp = layer.fc2(torch.nn.functional.gelu(layer.fc1(normed)))
            hidden = residual + mlp
            layer_outputs.append(hidden.clone())
        enc_final = encoder.layer_norm(hidden)
    for i in (0, 1, 5, 15, 31, len(layer_outputs) - 1):
        if i < len(layer_outputs):
            np.save(os.path.join(out_dir, f"4_layer_{i}.npy"), layer_outputs[i][0].numpy())
    np.save(os.path.join(out_dir, "5_encoder_output.npy"), enc_final[0].numpy())

    # ---- 6. projector -----------------------------------------------------
    stack_factor = 8
    eps = 1e-6
    ln_pre = sd["multi_modal_projector.ln_pre.weight"].to(torch.float32)
    w1 = sd["multi_modal_projector.linear_1.weight"].to(torch.float32)
    ln_mid = sd["multi_modal_projector.ln_mid.weight"].to(torch.float32)
    w2 = sd["multi_modal_projector.linear_2.weight"].to(torch.float32)

    def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    x = enc_final[0]  # [T2, 1280]
    t2 = x.shape[0]
    t_pad = (t2 + stack_factor - 1) // stack_factor * stack_factor
    stacked = torch.nn.functional.pad(x, (0, 0, 0, t_pad - t2))
    stacked = stacked.view(t_pad // stack_factor, -1)  # [T8, 10240]
    h = rms_norm(stacked, ln_pre) @ w1.T
    value, gate = h.chunk(2, dim=-1)
    h = torch.nn.functional.silu(gate) * value
    h = rms_norm(h, ln_mid) @ w2.T
    np.save(os.path.join(out_dir, "6_projector_output.npy"), h.numpy())

    # ---- 7. text + merge ----------------------------------------------------
    # render the chat template FIRST (same constant ember uses), then split
    # the rendered text around the placeholder
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    rendered = TEMPLATE.format(content=prompt)
    parts = rendered.split("<|audio|>")
    assert len(parts) == 2, "single-placeholder validation"
    n_audio_tokens = (n_frames + 15) // 16  # ceil(frames / (ds=2 * stack=8))
    ids: list[int] = []
    start = None
    for i, part in enumerate(parts):
        ids.extend(tok(part, add_special_tokens=False)["input_ids"])
        if i == 0:
            start = len(ids)
            ids.extend([eot] * n_audio_tokens)
    input_ids = torch.tensor([ids])

    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    embeds = model.get_input_embeddings()(input_ids)
    merged = embeds.clone()
    merged[0, start : start + n_audio_tokens] = h[:n_audio_tokens]
    np.save(os.path.join(out_dir, "7_assembled_embeddings.npy"), merged[0].detach().numpy())
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids[0].numpy())

    # ---- 8-9. prefill logits + greedy generation -----------------------------
    with torch.no_grad():
        out = model(inputs_embeds=merged, use_cache=True)
        past = out.past_key_values
        step_logits = [out.logits[0, -1].numpy()]
        gen_ids = [int(np.argmax(step_logits[-1]))]
        cur = torch.tensor([[gen_ids[-1]]])
        pos_offset = merged.shape[1]
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
            pos_offset += 1
    np.save(os.path.join(out_dir, "8_first_logits.npy"), step_logits[0])
    np.save(os.path.join(out_dir, "step_logits.npy"), np.stack(step_logits))
    gen_text = tok.decode(gen_ids, skip_special_tokens=True)

    manifest = {
        "model": MODEL_DIR,
        "ultravox_safetensors": ULTRAVOX_SAFETENSORS,
        "audio": wav_path,
        "prompt": prompt,
        "max_new_tokens": max_new,
        "input_ids_len": int(input_ids.shape[1]),
        "n_frames": int(n_frames),
        "n_audio_tokens": int(n_audio_tokens),
        "audio_start": int(start),
        "generation_ids": gen_ids,
        "generated_text": gen_text,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fjson:
        json.dump(manifest, fjson, indent=2)
    print("wrote", out_dir)
    print("input_ids_len:", input_ids.shape[1], "audio tokens:", n_audio_tokens, "at", start)
    print("generation ids:", gen_ids)
    print("generated:", repr(gen_text))


if __name__ == "__main__":
    main()
