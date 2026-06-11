# LiTo Studio

**One photo in. A 3D gaussian splat and a triangle mesh out. Entirely on your Mac.**

![macOS 15+](https://img.shields.io/badge/macOS-15%2B-blue) ![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-required-orange) ![Swift 6](https://img.shields.io/badge/Swift-6.0-F05138) ![License: MIT](https://img.shields.io/badge/license-MIT-green)

LiTo Studio is a native macOS app that runs Apple's **LiTo** image-to-3D model fully on-device with [MLX-Swift](https://github.com/ml-explore/mlx-swift). No Python, no server, no cloud: the complete DINOv2 → diffusion-transformer → sparse-voxel → gaussian pipeline executes on your GPU, and the result renders in a true 3DGS Metal viewer.

## Features

- **Full native pipeline** — DINOv2 conditioning, flow-matching DiT sampling with classifier-free guidance, sparse-voxel decoding, and a gaussian decoder with SH3 view-dependent color, all in Swift on MLX.
- **CoreML preprocessing** — RMBG 2.0 background removal (with an Apple Vision fallback), Real-ESRGAN 4× upscaling, automatic low-light normalization, and person-mask trimming.
- **Live feedback** — an intermediate occupancy point cloud renders while the sampler runs, plus a preview strip of every preprocessing stage.
- **Three viewers** — a Metal 3DGS splat renderer ([MetalSplatter](https://github.com/scier/MetalSplatter)), a SceneKit mesh viewer, and a SceneKit point-cloud viewer, switchable per result.
- **Quality tooling** — best-of-N seed search scored by silhouette IoU, marching-cubes mesh extraction, and optional photo-measured normal refinement (Meta Sapiens).
- **Auto-tuned settings** — the app analyzes each input image and recommends steps, guidance scale, and thresholds; everything stays manually adjustable.

## How it works

| Stage | What happens | Code |
|---|---|---|
| 1. Preprocess | Low-light normalize → Real-ESRGAN 4× → RMBG 2.0 cutout → person trim | `Sources/LiToKit/Preprocess.swift`, `RMBG.swift`, `Upscaler.swift` |
| 2. Condition | DINOv2 encodes the 518² RGBA cutout | `Sources/LiToKit/Dinov2.swift` |
| 3. Sample | DiT flow sampling (default 20 steps, CFG 3.0) over the sparse-structure latent | `Sources/LiToKit/Dit.swift` |
| 4. Decode | Sparse-voxel VAE decode → occupied voxel grid | `Sources/LiToKit/Trellis.swift` |
| 5. Splat | Gaussian decoder emits position/scale/rotation/opacity/SH3 per splat | `Sources/LiToKit/GaussianDecoder.swift` |
| 6. Mesh | Marching cubes over the gaussian density field, optional Sapiens normal refinement | `Sources/LiToKit/MeshExtract.swift`, `NormalRefine.swift` |

Every generation writes three artifacts:

- `*_gs.ply` — the primary output: a standard 3DGS gaussian splat (full SH3 color, per-splat scale/rotation/opacity). Opens in any 3DGS viewer (SuperSplat, etc.). Coordinates are LiTo-native **z-up**.
- `*_mesh.ply` / `*_mesh.obj` — a marching-cubes surface with vertex colors; opens in Blender or Preview.
- `*_pc.ply` — a colored point cloud for quick previews.

The app saves results to `~/Library/Application Support/LiToStudio/results/`.

## Requirements

- Apple Silicon Mac (16 GB unified memory or more recommended)
- macOS 15+ (developed on macOS 26)
- Xcode 16+ (Swift 6 toolchain and the Metal compiler)
- ~8 GB of model weights (see below)

## Quick start

```bash
git clone https://github.com/MenuuTUX/LiToStudio.git
cd LiToStudio
./run.sh
```

That's it. On a machine without the models, the app opens a **first-run setup screen**: one click downloads everything (~7.8 GB, checksum-verified, resumable) and converts Apple's LiTo checkpoint locally — then drops you into the app. Plan for ~15 GB of free disk during setup (the 7.4 GB checkpoint and its converted form briefly coexist; the checkpoint is deleted afterwards).

`run.sh` builds with SwiftPM, compiles the splat-viewer shaders, colocates the MLX Metal library, and launches. See [SETUP.md](SETUP.md) for the full developer setup, including running from Xcode.

### What first-run setup installs

| File | Required | Source |
|---|---|---|
| `lito.safetensors` (~7.4 GB) | yes | [Apple's LiTo checkpoint](https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt), downloaded and converted on your Mac by the bundled pure-Swift converter |
| `ss_*_conv3d_16l8_fp16.safetensors` + `.json` | yes | Sparse-voxel tokenizer, fetched from [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large) (byte-identical to the files the LiTo release uses; sha256-pinned) |
| `mlx.metallib` | yes | MLX's compiled GPU kernel library (MIT), from this repo's [releases](https://github.com/MenuuTUX/LiToStudio/releases) |
| `RealESRGAN_x4.mlmodel` | optional | CoreML conversion of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 4× (BSD-3), from this repo's releases |

Two optional models are **not** auto-downloaded because their licenses don't permit redistribution — the app works without them:

- `RMBG2.mlpackage` ([BriaAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) background removal) — without it the pipeline falls back to Apple Vision.
- `SapiensNormal.mlpackage` (Meta Sapiens normal estimator for photo-measured mesh refinement) — produce it with `./tools/convert_sapiens2/convert.sh` or the bundled Colab notebook.

Weights live in `weights/` next to the checkout (or `~/Library/Application Support/LiToStudio/weights` for a bare .app). Point anywhere else with `LITO_WEIGHTS_DIR=/path/to/weights`. Prefer doing it by hand or offline? Every manual step is in [SETUP.md](SETUP.md).

## Using the app

1. **Drop a photo in.** The app analyzes it and pre-fills recommended settings (steps, CFG, thresholds); toggle *Auto* off to set them yourself.
2. **Generate.** A live occupancy cloud appears while the sampler runs, and the preview strip shows each preprocessing stage (input → low-light fix → upscale → cutout).
3. **Inspect.** Switch between **Splat** (Metal 3DGS renderer), **Mesh**, and **Points** views; drag to orbit, scroll to zoom.
4. Results accumulate in `~/Library/Application Support/LiToStudio/results/`; the app can also open existing `.ply` / `.obj` / `.usdz` / `.stl` files.

Best-of-N seed search (`Seed candidates` setting) generates N candidates and keeps the one whose silhouette best matches the photo — slower, noticeably better geometry.

## Command line

`run.sh` wraps the most useful paths:

```bash
./run.sh                  # build (release) + launch the app
./run.sh smoke            # MLX + weights sanity check, no UI
./run.sh engine IMG out.ply 30          # full pipeline → point cloud + splat
./run.sh sculpt IMG out                 # photo → splat → refined mesh (quality path)
```

The `LiToSmoke` binary underneath exposes more for development — per-stage checks (`dino`, `dit`, `voxel`, `gs`, `rmbg`, `cond`), `mesh` re-extraction from any splat, `refine`/`normals` for Sapiens, `render` for headless PLY renders, and `score`/`gscheck` diagnostics. See [SETUP.md](SETUP.md) for the full argument lists.

## Benchmarks and fine-tuning

- `bench/run_baseline.sh` runs the quality protocol (25 steps, CFG 3.0, seed 7, best-of-3, mesh + render per photo; roughly 1.5–2.5 h per photo on a 16 GB machine). Compare every quality-affecting change against it — single-photo spot checks lie. Findings log: [bench/EXPERIMENTS.md](bench/EXPERIMENTS.md).
- [docs/FINETUNE_HUMANS.md](docs/FINETUNE_HUMANS.md) describes fine-tuning the DiT for clothed-human subjects on THuman2.1/2K2K (research-only datasets), with a ready-to-run Colab notebook in `docs/`.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `MLX error: Failed to load the default metallib` | `mlx.metallib` must sit next to the executable. `./run.sh`, the Xcode post-build step, and the app itself (after first-run setup) all handle this; a bare `swift build` binary won't have it. |
| Splat view missing or empty | The MetalSplatter shader library wasn't compiled (plain `swift build` doesn't compile Metal resources). `./run.sh` builds it automatically; Xcode builds always have it. |
| "Model weights not found" / setup reappears | Run first-run setup to completion, check `weights/lito.safetensors` exists, or set `LITO_WEIGHTS_DIR`. |
| A download fails mid-setup | Hit Retry — downloads resume from where they stopped, and every file is checksum-verified before install. |
| Very slow / memory pressure | The model holds ~7.4 GB; on 16 GB Macs close other heavy apps, lower the step count, and disable best-of-N. The first generation also pays a one-time model load. |

## Acknowledgements

- [Apple ml-lito](https://github.com/apple/ml-lito) — the LiTo model and reference implementation
- [MLX / MLX-Swift](https://github.com/ml-explore/mlx-swift) — the array framework powering the engine
- [MetalSplatter](https://github.com/scier/MetalSplatter) — 3DGS rendering and splat I/O
- [BriaAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [Meta Sapiens](https://github.com/facebookresearch/sapiens2) — preprocessing and refinement models
- [ComfyUI-LiTo](https://github.com/PozzettiAndrea/ComfyUI-LiTo) — reference implementation used for output parity

## License

The code is released under the [MIT License](LICENSE). Model weights are **not** covered: LiTo, RMBG-2.0 (commercial use requires a Bria license), Real-ESRGAN, and Sapiens each keep their upstream licenses — review them before shipping anything built on this.
