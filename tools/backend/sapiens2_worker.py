#!/usr/bin/env python3
"""Sapiens2 human pose / feature worker (facebook/sapiens2-pose-0.4b, ungated).

argv[1] = manifest JSON:
  {
    "views": [{"index": 0, "label": "front", "image": "/abs.png",
               "subjectBox": [x, y, w, h] | null}],   # normalized; person crop hint
    "outDir": "/abs/dir"
  }
stdout = one JSON document:
  {"ok": true, "backend": "...", "views": [
     {"viewIndex": 0, "personBox": [x,y,w,h], "keypointCount": 308,
      "keypointsFile": "/abs/v1_pose.json",
      "groups": {"body": {"visible": 17, "total": 23, "meanConfidence": 0.81}, ...},
      "raisedHand": "right" | "left" | "both" | null,
      "headY": 0.12}]}
Full per-keypoint output (name, xy normalized to the original image, score) goes to
<outDir>/v{i}_pose.json so the app stays lean.

Sapiens2 pose is a top-down model: it expects a person crop. The upstream
recommendation is an RTMDet person detector (facebook/sapiens-pose-bbox-detector,
mmdet). To keep the stack lean we instead use the REAL subject box estimated from
the RMBG cutout / border heuristic that the app already computes (passed as
subjectBox); full frame is the fallback. This is honest but slightly looser than
RTMDet — documented in LITO_TECHNICAL_NOTES.
"""
import sys, os, json, traceback

def log(*a):
    print(*a, file=sys.stderr, flush=True)

def fail(error, instruction):
    print(json.dumps({"ok": False, "error": str(error), "instruction": instruction}))
    sys.exit(0)

MODEL = "facebook/sapiens2-pose-0.4b"
VIS_THRESHOLD = 0.3

def group_of(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("thumb", "index", "middle", "ring", "pinky", "palm")):
        return "hands"
    if any(k in n for k in ("eye", "nose", "mouth", "lip", "ear", "brow", "face", "jaw", "cheek", "chin", "forehead", "teeth", "tongue", "iris", "pupil", "eyelid")):
        return "face"
    if any(k in n for k in ("ankle", "heel", "toe", "foot")):
        return "feet"
    return "body"

def main():
    manifest = json.load(open(sys.argv[1]))
    views = manifest["views"]
    out_dir = manifest["outDir"]
    os.makedirs(out_dir, exist_ok=True)

    import torch
    import numpy as np
    from PIL import Image

    try:
        from transformers import Sapiens2ForPoseEstimation, AutoProcessor
    except ImportError as e:
        fail(e, "transformers in tools/backend/.venv lacks Sapiens2 — rerun tools/backend/setup.sh")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    log(f"loading {MODEL} on {device}…")
    try:
        model = Sapiens2ForPoseEstimation.from_pretrained(MODEL, torch_dtype=torch.float32).to(device).eval()
        processor = AutoProcessor.from_pretrained(MODEL)
    except Exception as e:
        fail(e, f"could not load {MODEL} — check network / `hf download {MODEL}`")

    id2label = getattr(model.config, "id2label", {}) or {}
    # The HF config ships generic LABEL_n names; the canonical Goliath-308 table
    # (the 344 Goliath keypoints minus the 36 teeth keypoints, order preserved —
    # sapiens pose/configs/_base_/datasets/goliath.py, remove_teeth=True) lives next
    # to this script as goliath_keypoints.json.
    goliath_names = None
    table = os.path.join(os.path.dirname(os.path.abspath(__file__)), "goliath_keypoints.json")
    if os.path.exists(table):
        goliath_names = json.load(open(table))
        log(f"loaded goliath keypoint table ({len(goliath_names)} names)")
    import transformers
    backend = f"{MODEL} via transformers {transformers.__version__} ({device})"

    results = []
    for view in views:
        img = Image.open(view["image"]).convert("RGB")
        W, H = img.size
        sb = view.get("subjectBox")
        if sb:  # normalized xywh -> absolute COCO xywh, padded 5%
            x, y, w, h = sb[0] * W, sb[1] * H, sb[2] * W, sb[3] * H
            pad_x, pad_y = 0.05 * w, 0.05 * h
            box = [max(0, x - pad_x), max(0, y - pad_y),
                   min(W, w + 2 * pad_x), min(H, h + 2 * pad_y)]
        else:
            box = [0.0, 0.0, float(W), float(H)]

        inputs = processor(img, boxes=[[box]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        pose = processor.post_process_pose_estimation(outputs, boxes=[[box]])[0][0]
        kps = pose["keypoints"].cpu().numpy()       # (K, 2) absolute pixel coords
        scores = pose["scores"].cpu().numpy()       # (K,)
        labels = pose.get("labels")
        K = len(scores)

        names = []
        for i in range(K):
            idx = int(labels[i]) if labels is not None else i
            if goliath_names and idx < len(goliath_names) and K == len(goliath_names):
                names.append(goliath_names[idx])
            else:
                n = str(id2label.get(idx, id2label.get(str(idx), f"kp_{idx}")))
                names.append(goliath_names[idx] if (goliath_names and n.startswith("LABEL_")
                                                    and idx < len(goliath_names)) else n)

        # Full per-keypoint artifact (normalized coords).
        full = [{"name": names[i],
                 "x": float(kps[i][0] / W), "y": float(kps[i][1] / H),
                 "score": float(scores[i])} for i in range(K)]
        kp_file = os.path.join(out_dir, f"v{view['index'] + 1}_pose.json")
        json.dump({"backend": backend, "keypoints": full}, open(kp_file, "w"))

        # Group summaries.
        groups = {}
        for g in ("body", "face", "hands", "feet"):
            idxs = [i for i in range(K) if group_of(names[i]) == g]
            vis = [i for i in idxs if scores[i] >= VIS_THRESHOLD]
            groups[g] = {
                "visible": len(vis), "total": len(idxs),
                "meanConfidence": float(np.mean([scores[i] for i in vis])) if vis else 0.0,
            }

        # Derived pose cues from real keypoints (image y grows downward).
        def find(sub):
            c = [i for i in range(K) if sub in names[i].lower() and scores[i] >= VIS_THRESHOLD]
            return min(c, key=lambda i: -scores[i]) if c else None
        nose = find("nose")
        lw, rw = find("left_wrist"), find("right_wrist")
        raised = None
        if nose is not None:
            ny = kps[nose][1]
            lup = lw is not None and kps[lw][1] < ny
            rup = rw is not None and kps[rw][1] < ny
            raised = "both" if (lup and rup) else "left" if lup else "right" if rup else None

        results.append({
            "viewIndex": view["index"],
            "personBox": [box[0] / W, box[1] / H, box[2] / W, box[3] / H],
            "keypointCount": K,
            "keypointsFile": kp_file,
            "groups": groups,
            "raisedHand": raised,
            "headY": float(kps[nose][1] / H) if nose is not None else None,
        })
        log(f"  v{view['index'] + 1}: {K} keypoints, "
            + ", ".join(f"{g} {v['visible']}/{v['total']}" for g, v in groups.items()))

    print(json.dumps({"ok": True, "backend": backend, "views": results}))

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": str(e), "instruction": "see stderr traceback"}))
