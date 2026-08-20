#!/usr/bin/env python3
"""Capture reference SmolVLM-256M-Instruct activations from transformers.

Validates the ember multimodal pipeline against the upstream reference at
every boundary:
  1. processed pixel tensor          (pixel_values)
  2. patch embeddings                (vision embeddings output)
  3. vision layer outputs            (selected layers)
  4. vision encoder output           (post_layernorm)
  5. projector output                (connector / modality projection)
  6. assembled LLM input embeddings  (inputs_merger result)
  7. first LLM logits                (prefill logits at last position)
  8. short greedy generation         (16 tokens, fp32)

Usage:
  python scripts/ref_smolvlm.py <image> <prompt> <out_dir> [max_new_tokens]

Writes <out_dir>/*.npy + manifest.json. The model runs in fp32 on CPU.
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> None:
    image_path, prompt, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    max_new = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    os.makedirs(out_dir, exist_ok=True)

    proc = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt")

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    pixel_mask = inputs["pixel_attention_mask"]

    # ---- vision encoder intermediates ---------------------------------
    vision = model.model.vision_model
    patch_mask = torch.ones(pixel_values.shape[0] * pixel_values.shape[1],
                            pixel_values.shape[3] // 16, pixel_values.shape[4] // 16,
                            dtype=torch.bool)
    embeds = vision.embeddings(pixel_values=pixel_values.view(-1, 3, 512, 512),
                               patch_attention_mask=patch_mask)
    layer_outputs = []
    hidden = embeds
    for layer in vision.encoder.layers:
        hidden = layer(hidden, attention_mask=None)
        layer_outputs.append(hidden.detach().numpy())
    encoder_out = vision.post_layernorm(hidden)

    # ---- connector -----------------------------------------------------
    features = model.model.connector(encoder_out)

    # ---- text embeddings + merge --------------------------------------
    text_embeds = model.model.text_model.get_input_embeddings()(input_ids)
    merged = model.model.inputs_merger(input_ids=input_ids,
                                       inputs_embeds=text_embeds,
                                       image_hidden_states=features)

    # ---- LLM prefill (first logits) -----------------------------------
    with torch.no_grad():
        out = model(input_ids=input_ids, pixel_values=pixel_values,
                    pixel_attention_mask=pixel_mask, use_cache=True,
                    logits_to_keep=1)
    first_logits = out.logits[0, -1].detach().numpy()

    # ---- greedy generation with per-step logits -------------------------
    past = None
    step_logits = []
    gen_ids = []
    cur_ids = input_ids
    with torch.no_grad():
        for _ in range(max_new):
            out = model(input_ids=cur_ids, pixel_values=pixel_values,
                        pixel_attention_mask=pixel_mask, use_cache=True,
                        past_key_values=past)
            past = out.past_key_values
            logits = out.logits[0, -1].numpy()
            step_logits.append(logits)
            nxt = int(np.argmax(logits))
            gen_ids.append(nxt)
            cur_ids = torch.tensor([[nxt]])
    gen_text = proc.tokenizer.decode(gen_ids, skip_special_tokens=True)
    np.save(os.path.join(out_dir, "step_logits.npy"), np.stack(step_logits))

    def save(name, tensor):
        np.save(os.path.join(out_dir, name), tensor.detach().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor))

    save("1_pixels.npy", pixel_values[0])
    save("2_patch_embeddings.npy", embeds)
    for i in [0, 1, 5, 11]:
        save(f"3_layer_{i}.npy", layer_outputs[i])
    save("4_encoder_output.npy", encoder_out)
    save("5_projector_output.npy", features)
    save("6_assembled_embeddings.npy", merged[0])
    save("7_first_logits.npy", first_logits)
    save("8_generation_ids.npy", np.array(gen_ids, dtype=np.int64))
    np.save(os.path.join(out_dir, "input_ids.npy"), input_ids[0].numpy())
    np.save(os.path.join(out_dir, "text_embeddings.npy"), text_embeds[0].detach().numpy())

    manifest = {
        "model": MODEL,
        "image": image_path,
        "prompt": prompt,
        "text": text,
        "max_new_tokens": max_new,
        "input_ids_len": int(input_ids.shape[1]),
        "n_images": int(pixel_values.shape[1]),
        "tile_grid": None,
        "gen_text": gen_text,
        "gen_ids": gen_ids,
        "pixel_values_shape": list(pixel_values.shape),
        "vocab_size": int(model.config.text_config.vocab_size),
        "hidden_size": int(model.config.text_config.hidden_size),
        "generated_text": gen_text,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("wrote", out_dir)
    print("text:", repr(text))
    print("generation ids:", gen_ids)
    print("generated:", repr(gen_text))


if __name__ == "__main__":
    main()
