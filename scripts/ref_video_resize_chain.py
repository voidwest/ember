#!/usr/bin/env python3
"""Phase 5 Track H: capture the STOCK SmolVLM2-video preprocessing chain
(with resizing enabled — the piece Phase 3 had to neutralize).

Wraps torchvision's functional resize + the processor's resize call site
to record every (input shape, out size, mode, antialias) actually applied,
plus dumps the intermediate tensors so ember can port the arithmetic
exactly and pin it bit-comparably.

Usage: python scripts/ref_video_resize_chain.py <frames_dir> <out_dir> [fps]
"""
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

MODEL = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def main() -> None:
    frames_dir, out_dir = sys.argv[1], sys.argv[2]
    src_fps = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0
    os.makedirs(out_dir, exist_ok=True)

    names = sorted(n for n in os.listdir(frames_dir) if n.endswith(".png"))
    pil_frames = [Image.open(os.path.join(frames_dir, n)).convert("RGB") for n in names]
    print(f"{len(pil_frames)} frames of {pil_frames[0].size}")

    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)

    # ---- wrap the image processor's resize to record the chain ----------
    ip = proc.video_processor if hasattr(proc, "video_processor") else proc.image_processor
    orig_resize = ip.resize.__func__

    calls = []

    def traced_resize(self, image, size, **kwargs):
        rec = {
            "in_shape": list(getattr(image, "shape", getattr(image, "size", None)))
            if not isinstance(image, Image.Image)
            else list(image.size),
            "size_arg": repr(size),
            "kwargs": {k: str(v) for k, v in kwargs.items()},
        }
        out = orig_resize(self, image, size, **kwargs)
        rec["out_shape"] = (
            list(out.shape) if torch.is_tensor(out) else list(out.size)
        )
        calls.append(rec)
        return out

    import types

    ip.resize = types.MethodType(traced_resize, ip)

    from transformers.video_utils import VideoMetadata

    n_frames = len(pil_frames)
    meta = VideoMetadata(
        total_num_frames=n_frames,
        fps=src_fps,
        width=pil_frames[0].width,
        height=pil_frames[0].height,
        duration=n_frames / src_fps,
        frames_indices=list(range(n_frames)),
    )
    user_text = "What happens in this video?"
    messages = [
        {
            "role": "user",
            "content": [{"type": "video"}, {"type": "text", "text": user_text}],
        }
    ]
    text = proc.apply_chat_template(messages, add_generation_prompt=True)
    inputs = proc(
        text=[text],
        videos=[[pil_frames]],
        videos_kwargs={
            "do_sample_frames": False,
            "video_metadata": [[meta]],
        },
        return_tensors="pt",
    )
    pv = inputs["pixel_values"]
    print("pixel_values:", tuple(pv.shape))
    for c in calls:
        print("resize:", c)

    manifest = {
        "model": MODEL,
        "frames_dir": frames_dir,
        "resize_calls": calls,
        "pixel_values_shape": list(pv.shape),
    }
    with open(os.path.join(out_dir, "resize_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)

    # dump first frame through each recorded stage for bit-level porting
    pv_frames = pv.view(-1, *pv.shape[-3:])
    np.save(os.path.join(out_dir, "pixels_frame0.npy"), pv_frames[0].numpy())
    np.save(
        os.path.join(out_dir, "input_ids.npy"),
        inputs["input_ids"][0].numpy(),
    )

    # ---- reproduce the chain manually for the FIRST frame ---------------
    # (whatever the recorded sequence is, apply it step by step in torch so
    # ember can be compared against each stage independently)
    import torchvision.transforms.functional as F

    img = pil_frames[0]
    stage = 0
    cur_np = np.array(img)
    for c in calls[:4]:
        size = c["size_arg"]
        # only handle the common dict forms
        try:
            parsed = eval(size)  # noqa: S307 - trusted local file
        except Exception:
            continue
        t = torch.from_numpy(cur_np).permute(2, 0, 1).float() / 255.0
        if isinstance(parsed, dict):
            if "longest_edge" in parsed:
                le = parsed["longest_edge"]
                h, w = t.shape[-2:]
                scale = le / max(h, w)
                nh, nw = round(h * scale), round(w * scale)
                t2 = F.resize(t, [nh, nw], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
            elif "shortest_edge" in parsed:
                se = parsed["shortest_edge"]
                h, w = t.shape[-2:]
                scale = se / min(h, w)
                nh, nw = round(h * scale), round(w * scale)
                t2 = F.resize(t, [nh, nw], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
            else:
                wh = parsed.get("height"), parsed.get("width")
                if wh[0] and wh[1]:
                    t2 = F.resize(t, [wh[0], wh[1]], interpolation=F.InterpolationMode.BICUBIC, antialias=True)
                else:
                    continue
        elif isinstance(parsed, int):
            t2 = F.resize(t, parsed, interpolation=F.InterpolationMode.BICUBIC, antialias=True)
        else:
            continue
        np.save(os.path.join(out_dir, f"stage{stage}_manual.npy"), t2.numpy())
        cur_np = (t2.numpy().transpose(1, 2, 0) * 255.0).round().clip(0, 255).astype(np.uint8)
        stage += 1
    print(f"dumped {stage} manual stages")
    print("wrote", out_dir)


if __name__ == "__main__":
    main()
