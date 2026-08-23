#!/usr/bin/env python3
"""SmolVLM2-256M-Video-Instruct reference capture (video path).

Feeds a directory of PNG frames as one pre-sampled video through the HF
processor + Idefics3ForConditionalGeneration (fp32 CPU) and captures every
boundary ember's video path also exposes:

  1_pixels            post-resize normalized frames [n,3,512,512]
  4_encoder_output    vision tower output [n*1024,768]
  5_projector_output  connector output [n*64,576]
  6_assembled         merged LLM input embeddings [seq,576]
  7_first_logits      prefill last-position logits
  step_logits + gen_ids

Usage:
  python scripts/ref_smolvlm2_video.py <frames_dir> "<prompt with <video>>" \
      <out_dir> [max_new_tokens] [--source-fps F]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src_fps = 24.0
    if "--source-fps" in sys.argv:
        src_fps = float(sys.argv[sys.argv.index("--source-fps") + 1])
    frames_dir, prompt, out_dir = args[0], args[1], args[2]
    max_new = int(args[3]) if len(args) > 3 else 16
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))
    pil_frames = [Image.open(os.path.join(frames_dir, n)).convert("RGB") for n in names]
    imgs = [np.array(f) for f in pil_frames]
    video = pil_frames  # PIL frames keep the PIL/LANCZOS path
    print(f"{len(imgs)} frames of {imgs[0].shape[:2]}")

    # PIL backend: keeps LANCZOS resize (the torchvision fast path silently
    # falls back to BICUBIC on tensor input)
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32).eval()

    # render the chat template exactly like ref_smolvlm.py does for images
    user_text = prompt.removeprefix("<video>")
    messages = [{"role": "user", "content": [
        {"type": "video"},
        {"type": "text", "text": user_text},
    ]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    print("template:", repr(text[:120]))

    # explicit metadata so frame timestamps match the declared source fps
    from transformers.video_utils import VideoMetadata
    n_frames = len(video)
    meta = VideoMetadata(total_num_frames=n_frames, fps=src_fps,
                         width=video[0].width, height=video[0].height,
                         duration=n_frames / src_fps,
                         frames_indices=list(range(n_frames)))
    # NOTE: the stock video processor inherits size {"longest_edge": 2048}
    # from the image kwargs, so frames get upsampled to 2048 and back down
    # to 512 (bicubic both times). Phase 5 Track H closes the parity debt:
    # pass --stock-resize to run the STOCK chain (no do_resize override);
    # ember reproduces that chain with PIL-exact bicubic and is validated
    # against this reference WITHOUT neutralization.
    videos_kwargs = {
        "do_sample_frames": False,
        "video_metadata": [[meta]],
    }
    if "--stock-resize" not in sys.argv:
        videos_kwargs["do_resize"] = False
    inputs = proc(text=[text], videos=[[video]],
                  videos_kwargs=videos_kwargs,
                  return_tensors="pt")
    input_ids = inputs["input_ids"]
    pv = inputs["pixel_values"]
    print("pixel_values:", tuple(pv.shape))
    expanded = proc.tokenizer.decode(input_ids[0])
    print("expanded head:", repr(expanded[:200]))

    with open(os.path.join(out_dir, "expanded_text.txt"), "w") as f:
        f.write(proc.tokenizer.decode(input_ids[0]))
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids[0].numpy())
    np.save(os.path.join(out_dir, "1_pixels.npy"),
            pv.view(-1, *pv.shape[-3:]).numpy())

    # ---- vision tower over all frames in one batch -----------------------
    vision = model.model.vision_model
    n_frames = pv.shape[1]
    patch_mask = torch.ones(n_frames, 512 // 16, 512 // 16, dtype=torch.bool)
    embeds = vision.embeddings(pixel_values=pv.view(-1, 3, 512, 512),
                               patch_attention_mask=patch_mask)
    hidden = embeds
    with torch.no_grad():
        for layer in vision.encoder.layers:
            hidden = layer(hidden, attention_mask=None)
        enc_out = vision.post_layernorm(hidden)
    np.save(os.path.join(out_dir, "4_encoder_output.npy"), enc_out.numpy())

    features = model.model.connector(enc_out)  # [n_frames, 64, 576]
    np.save(os.path.join(out_dir, "5_projector_output.npy"),
            features.reshape(-1, features.shape[-1]).detach().numpy())

    # ---- merge ------------------------------------------------------------
    tokz = proc.tokenizer
    embed_layer = model.get_input_embeddings()
    embeds = embed_layer(input_ids)
    image_id = tokz.convert_tokens_to_ids("<image>")
    mask = input_ids[0] == image_id
    n_img_tokens = int(mask.sum())
    features_flat = features.reshape(-1, features.shape[-1])
    assert n_img_tokens == features_flat.shape[0], (n_img_tokens, features_flat.shape)
    merged = embeds.clone()
    merged[0, mask] = features_flat.detach().to(merged.dtype)
    np.save(os.path.join(out_dir, "6_assembled.npy"), merged[0].detach().numpy())

    # ---- generate ----------------------------------------------------------
    with torch.no_grad():
        out = model(inputs_embeds=merged, use_cache=True)
        past = out.past_key_values
        steps = [out.logits[0, -1].numpy()]
        gen = [int(np.argmax(steps[-1]))]
        cur = torch.tensor([[gen[-1]]])
        eos = {tokz.eos_token_id}
        for extra in ("<end_of_utterance>",):
            t = tokz.convert_tokens_to_ids(extra)
            if t is not None and t >= 0:
                eos.add(t)
        seq_len = merged.shape[1]
        for _ in range(max_new - 1):
            if gen[-1] in eos:
                break
            o = model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            lg = o.logits[0, -1].numpy()
            steps.append(lg)
            nxt = int(np.argmax(lg))
            gen.append(nxt)
            cur = torch.tensor([[nxt]])
            _ = seq_len
    np.save(os.path.join(out_dir, "7_first_logits.npy"), steps[0])
    np.save(os.path.join(out_dir, "step_logits.npy"), np.stack(steps))
    manifest = {
        "model": MODEL,
        "frames": len(imgs),
        "frame_size": list(imgs[0].shape[:2]),
        "source_fps": src_fps,
        "prompt": prompt,
        "input_ids_len": int(input_ids.shape[1]),
        "n_image_tokens": n_img_tokens,
        "gen_ids": gen,
        "generated_text": tokz.decode(gen),
        "pixel_values_shape": list(pv.shape),
    }
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    print("gen:", manifest["generated_text"])


if __name__ == "__main__":
    main()
