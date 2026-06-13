# LiTo Studio model backend (Python workers)

One `uv` venv that runs the heavy model backends as separate processes. The Swift app
calls them through `Sources/LiToKit/PythonBackend.swift` (manifest JSON in → one JSON
document out + image/JSON artifacts); each worker exits before MLX sampling starts, so
their memory never overlaps generation on 16 GB machines.

## Install

```bash
cd tools/backend
./setup.sh          # uv venv (python 3.12) + torch/transformers/coremltools etc.
./setup.sh check    # status of venv + each model
```

## Models

| Model | Status check | Install |
|---|---|---|
| **RMBG-2.0** (background removal) | `./setup.sh check` → `models--briaai--RMBG-2.0` | Accept the license at <https://huggingface.co/briaai/RMBG-2.0>, `hf auth login`, then `hf download briaai/RMBG-2.0`. Commercial use requires a Bria agreement. |
| **SAM 3.1 CoreML** (landmark grounding — PREFERRED, native, no gate) | `weights/sam3-coreml/` exists | `hf download AllanVester/SAM3.1-CoreML-FP16 --local-dir ../../weights/sam3-coreml` then `.venv/bin/python make_sam3_tokens.py` (bakes CLIP-BPE prompt tokens). Driver: `Sources/LiToKit/Sam3CoreML.swift`; no Python at runtime. See Decision 012 for empirical findings + license notes. |
| **SAM3 worker** (fallback / parity check) | `models--facebook--sam3` | **Gated, approval required:** request access at <https://huggingface.co/facebook/sam3>, wait for Meta's approval, then `hf download facebook/sam3`. The app prefers the CoreML backend when both exist. |
| **Sapiens2 pose 0.4b** (human pose / features) | `models--facebook--sapiens2-pose-0.4b` | `hf download facebook/sapiens2-pose-0.4b model.safetensors config.json preprocessor_config.json` (ungated, ~1.7 GB). The larger `facebook/sapiens2-pose-1b` works as a drop-in by editing `MODEL` in `sapiens2_worker.py` (more RAM/time). |

## Workers

- `rmbg_worker.py` — full-resolution RGBA cutouts (PyTorch/MPS). This is the ACTIVE
  background-removal path: converting BiRefNet to CoreML is blocked because it uses
  `torchvision::deform_conv2d`, which coremltools (≤ 9.x) cannot convert
  (`convert_rmbg2.py` is kept for when that changes; it currently fails at that op).
- `sam3_worker.py` — text-prompted concept segmentation per taxonomy token
  (`goliath`-neutral prompts live in `LandmarkGrounding.swift`); emits per-view
  boxes/masks/confidences + honest `not_detected` / `failed` statuses.
- `sapiens2_worker.py` — top-down 308-keypoint pose (Goliath-308 names in
  `goliath_keypoints.json`: the 344 Goliath keypoints minus 36 teeth, extracted from
  the sapiens repo's dataset config). Person crop comes from the app's real subject
  estimate instead of the upstream RTMDet detector (mmdet stack intentionally
  avoided); pass nothing and it uses the full frame.

## Standalone test (no engine weights needed)

```bash
swift build
.build/debug/LiToSmoke ground  <img1>[,<img2>…] /tmp/lito_ground          # all 3 backends
.build/debug/LiToSmoke sam3     weights/sam3-coreml <img> all 0.51 [cutout] # SAM3 core set
.build/debug/LiToSmoke sam3concept weights/sam3-coreml <img> "leather glove" [cutout]
```
`ground` prints per-backend status; cutouts/pose/masks/overlays land under the out dir.
SAM3 detections are gated by the person silhouette (pass a cutout for the cleanest
result) and rendered as region-over-photo overlays.

## Helper scripts
- `make_sam3_tokens.py` — bake the 12 taxonomy prompt token-IDs into
  `weights/sam3-coreml/prompt_tokens.json` (run once after downloading the SAM3 CoreML
  packages).
- `tokenize_clip.py` — CLIP-BPE tokenize free-text phrases (used by the app's text
  concept guidance for arbitrary user text; needs only the venv, no model download).
