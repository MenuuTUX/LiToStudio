#!/usr/bin/env python3
"""RMBG-2.0 cutout worker (PyTorch) — fallback path when the CoreML conversion of
BiRefNet is unavailable. Same honesty rules as the other workers: real model, real
masks; on any failure the JSON says so.

argv[1] = manifest JSON:
  {"views": [{"index": 0, "image": "/abs.png"}], "outDir": "/abs/dir"}
stdout:
  {"ok": true, "backend": "...", "views": [
     {"viewIndex": 0, "cutout": "/abs/v1_cutout.png", "width": W, "height": H}]}
Cutouts are full-resolution RGBA PNGs (premultiplied alpha from the predicted mask).
"""
import sys, os, json, traceback

def log(*a): print(*a, file=sys.stderr, flush=True)

def fail(error, instruction):
    print(json.dumps({"ok": False, "error": str(error), "instruction": instruction}))
    sys.exit(0)

def main():
    manifest = json.load(open(sys.argv[1]))
    views = manifest["views"]
    out_dir = manifest["outDir"]
    os.makedirs(out_dir, exist_ok=True)

    import torch
    import numpy as np
    from PIL import Image
    from transformers import AutoModelForImageSegmentation

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading briaai/RMBG-2.0 on {device}…")
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True).to(device).eval()
    except Exception as e:
        fail(e, "Accept the license at https://huggingface.co/briaai/RMBG-2.0 and `hf auth login`.")

    import transformers
    backend = f"briaai/RMBG-2.0 via transformers {transformers.__version__} ({device})"
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    results = []
    for view in views:
        img = Image.open(view["image"]).convert("RGB")
        W, H = img.size
        x = torch.from_numpy(np.array(img.resize((1024, 1024), Image.BILINEAR))).permute(2, 0, 1).float() / 255
        x = ((x - mean) / std).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(x)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            if hasattr(preds, "logits"):
                preds = preds.logits
            mask = torch.sigmoid(preds)[0, 0].cpu().numpy()
        # Sharpen the soft mask the same way RMBG.swift does (second sigmoid).
        mask = 1.0 / (1.0 + np.exp(-10.0 * (mask - 0.5)))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((W, H), Image.BILINEAR)
        rgba = img.convert("RGBA")
        rgba.putalpha(mask_img)
        out = os.path.join(out_dir, f"v{view['index'] + 1}_cutout.png")
        rgba.save(out)
        results.append({"viewIndex": view["index"], "cutout": out, "width": W, "height": H})
        log(f"  v{view['index'] + 1}: cutout {W}×{H}")

    print(json.dumps({"ok": True, "backend": backend, "views": results}))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e), "instruction": "see stderr traceback"}))
