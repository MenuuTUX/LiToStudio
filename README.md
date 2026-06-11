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
# put the weights in weights/ (next section), then:
./run.sh
```

`run.sh` builds with SwiftPM, compiles the splat-viewer shaders, colocates the MLX Metal library, points the engine at this checkout's `weights/`, and launches the app. See [SETUP.md](SETUP.md) for the full developer setup, including running from Xcode.

### Getting the weights

The repo ships **no weights**; place these in `weights/`:

| File | Required | What it is |
|---|---|---|
| `lito.safetensors` (~7.4 GB) | yes | DINOv2 + DiT + voxel VAE + gaussian decoder, converted from Apple's checkpoint (below) |
| `ss_dec_conv3d_16l8_fp16.safetensors` + `.json` (and the matching `ss_enc_*` pair) | yes | Sparse-voxel tokenizer from the [`apple/ml-lito`](https://github.com/apple/ml-lito) release (the engine reads the decoder at inference) |
| `mlx.metallib` | yes | MLX's compiled GPU kernel library — not produced by `swift build`; take it from an MLX build's products |
| `RMBG2.mlpackage` | no | CoreML conversion of [BriaAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) background removal; without it the app falls back to Apple Vision |
| `RealESRGAN_x4.mlmodel` | no | CoreML conversion of [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) 4× upscaler; skipped when absent |
| `SapiensNormal.mlpackage` | no | Meta Sapiens normal estimator for photo-measured mesh refinement; produce it with `./tools/convert_sapiens2/convert.sh` or the bundled Colab notebook |

Convert the main checkpoint with the bundled, dependency-free converter:

```bash
curl -L -o lito_dit_rgba.ckpt https://ml-site.cdn-apple.com/models/lito/lito_dit_rgba.ckpt
swift run -c release LiToConvert lito_dit_rgba.ckpt weights/lito.safetensors
```

To keep weights somewhere else, set `LITO_WEIGHTS_DIR=/path/to/weights`.

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
| `MLX error: Failed to load the default metallib` | `mlx.metallib` must sit next to the executable. `./run.sh` and the Xcode post-build step handle this; a bare `swift build` binary won't have it. |
| Splat view missing or empty | The MetalSplatter shader library wasn't compiled (plain `swift build` doesn't compile Metal resources). `./run.sh` builds it automatically; Xcode builds always have it. |
| "Model weights not found" | Check `weights/lito.safetensors` exists, or set `LITO_WEIGHTS_DIR`. |
| Very slow / memory pressure | The model holds ~7.4 GB; on 16 GB Macs close other heavy apps, lower the step count, and disable best-of-N. The first generation also pays a one-time model load. |

## Acknowledgements

- [Apple ml-lito](https://github.com/apple/ml-lito) — the LiTo model and reference implementation
- [MLX / MLX-Swift](https://github.com/ml-explore/mlx-swift) — the array framework powering the engine
- [MetalSplatter](https://github.com/scier/MetalSplatter) — 3DGS rendering and splat I/O
- [BriaAI RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0), [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), [Meta Sapiens](https://github.com/facebookresearch/sapiens2) — preprocessing and refinement models
- [ComfyUI-LiTo](https://github.com/PozzettiAndrea/ComfyUI-LiTo) — reference implementation used for output parity

## License

The code is released under the [MIT License](LICENSE). Model weights are **not** covered: LiTo, RMBG-2.0 (commercial use requires a Bria license), Real-ESRGAN, and Sapiens each keep their upstream licenses — review them before shipping anything built on this.
