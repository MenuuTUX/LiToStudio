#!/usr/bin/env python3
"""SAM3 landmark-grounding worker.

stdin/argv contract (the Swift adapter writes a manifest and reads stdout JSON):
  argv[1] = manifest JSON:
    {
      "views":   [{"index": 0, "label": "front", "image": "/abs/path.png"}],
      "prompts": [{"id": "L001", "token": "face_region", "phrase": "face"}],
      "outDir":  "/abs/dir/for/masks",
      "threshold": 0.4            # optional score threshold
    }
  stdout = one JSON document:
    {"ok": true, "backend": "...", "views": [
        {"viewIndex": 0, "viewLabel": "front", "landmarks": [
            {"id": "L001", "label": "face_region", "status": "detected",
             "bbox": [x, y, w, h],          # normalized, top-left origin
             "maskPath": "/abs/mask.png", "confidence": 0.93},
            {"id": "L004", "label": "abdomen_navel_region", "status": "not_detected"}
        ]}]}
  On failure: {"ok": false, "error": "...", "instruction": "..."} (exit 0 — the
  Swift side surfaces the message; non-zero exits are reserved for crashes).

Model: facebook/sam3 via transformers (promptable concept segmentation — text
phrase -> instance masks + boxes + scores). The repo is GATED: request access at
https://huggingface.co/facebook/sam3 and run `hf auth login` first.
Logs go to stderr; stdout carries exactly one JSON document.
"""
import sys, os, json, traceback

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def fail(error, instruction):
    print(json.dumps({"ok": False, "error": str(error), "instruction": instruction}))
    sys.exit(0)

def main():
    manifest = json.load(open(sys.argv[1]))
    views = manifest["views"]
    prompts = manifest["prompts"]
    out_dir = manifest["outDir"]
    threshold = float(manifest.get("threshold", 0.4))
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from PIL import Image
    import numpy as np

    try:
        from transformers import Sam3Processor, Sam3Model
    except ImportError as e:
        fail(e, "transformers in tools/backend/.venv has no SAM3 support — rerun tools/backend/setup.sh to upgrade transformers")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading facebook/sam3 on {device}…")
    try:
        model = Sam3Model.from_pretrained("facebook/sam3", torch_dtype=torch.float32).to(device).eval()
        processor = Sam3Processor.from_pretrained("facebook/sam3")
    except Exception as e:
        fail(e, "facebook/sam3 is gated: request access at https://huggingface.co/facebook/sam3 "
                "(Meta approval required), then `hf auth login` with the approved account and rerun.")

    import transformers
    backend = f"facebook/sam3 via transformers {transformers.__version__} ({device})"
    results = []
    for view in views:
        img = Image.open(view["image"]).convert("RGB")
        W, H = img.size
        landmarks = []
        for p in prompts:
            phrase = p.get("phrase") or p["token"].replace("_", " ")
            try:
                inputs = processor(images=img, text=phrase, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                post = getattr(processor, "post_process_instance_segmentation", None)
                res = post(outputs, threshold=threshold, mask_threshold=0.5,
                           target_sizes=[(H, W)])[0]
                scores = res["scores"].cpu().numpy() if len(res["scores"]) else np.zeros(0)
                if scores.size == 0 or float(scores.max()) < threshold:
                    landmarks.append({"id": p["id"], "label": p["token"], "status": "not_detected"})
                    continue
                best = int(scores.argmax())
                box = res["boxes"][best].cpu().numpy().tolist()      # xyxy absolute
                mask = res["masks"][best].cpu().numpy()
                mask_u8 = (mask.astype(np.uint8) * 255)
                mask_path = os.path.join(out_dir, f"v{view['index'] + 1}_{p['id']}.png")
                Image.fromarray(mask_u8, mode="L").save(mask_path)
                x0, y0, x1, y1 = box
                landmarks.append({
                    "id": p["id"], "label": p["token"], "status": "detected",
                    "bbox": [x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H],
                    "maskPath": mask_path,
                    "confidence": float(scores[best]),
                })
                log(f"  v{view['index'] + 1} {p['token']}: detected {scores[best]:.2f}")
            except Exception as e:
                log(f"  v{view['index'] + 1} {p['token']}: FAILED {e}")
                landmarks.append({"id": p["id"], "label": p["token"], "status": "failed",
                                  "error": str(e)[:200]})
        results.append({"viewIndex": view["index"], "viewLabel": view["label"],
                        "landmarks": landmarks})

    print(json.dumps({"ok": True, "backend": backend, "views": results}))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e), "instruction": "see stderr traceback"}))
