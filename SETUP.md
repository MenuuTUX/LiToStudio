# LiTo Studio — local setup

A 100% native (Swift + MLX, no Python) image → 3D gaussian-splat app for Apple Silicon macOS.

## Requirements
- Apple Silicon Mac, macOS 15+ (tested on macOS 26, Apple Silicon, 16 GB)
- Xcode 16+ (provides the Swift 6 toolchain and the Metal compiler)
- The model weights in `weights/` (not in the repo — place them there yourself):
  - `lito.safetensors` (~7.4 GB) — DINOv2 + DiT + voxel VAE + gaussian decoder
  - `ss_enc/dec_conv3d_16l8_fp16.safetensors` — voxel tokenizer
  - `mlx.metallib` — MLX's Metal shader library (**required**, see note below)
  - `RMBG2.mlpackage`, `RealESRGAN_x4.mlmodel` — CoreML pre-processing models

## Run it (simplest)

```bash
./run.sh            # build (release) + launch the app
./run.sh debug      # build (debug)   + launch the app
./run.sh smoke      # MLX + weights sanity check, no UI
./run.sh engine /path/to/image.png out.ply 30  # full pipeline → out.ply + out.gs.ply
./run.sh sculpt /path/to/image.png out         # photo → splat → refined mesh (quality path)
```

Every generation writes **three** artifacts:
- `*_pc.ply` — colored point cloud (quick preview; SceneKit / `LiToSmoke render`)
- `*_gs.ply` — the real output: a standard 3DGS gaussian splat (full SH3 view-dependent
  color, per-splat scale/rotation/opacity). The app renders it with a Metal splat
  renderer (MetalSplatter); it also loads in any 3DGS viewer (SuperSplat, etc.).
  Coordinates are LiTo-native **z-up** (viewers rotate; same convention as the reference).
- `*_mesh.ply` + `*_mesh.obj` — marching-cubes surface extracted from the gaussians
  (geometry-first artifact with vertex colors; opens in Blender/Preview).
  Re-extract from any splat with `LiToSmoke mesh in_gs.ply out [resolution] [iso]`.

`run.sh` builds with SwiftPM, copies `mlx.metallib` next to the binary, points the
engine at this checkout's `weights/`, and launches.

## Optional: Sapiens photo-measured mesh refinement

Drop `SapiensNormal.mlpackage` into `weights/` and mesh extraction (app and
`sculpt`/`refine` CLI) automatically re-sculpts the camera-facing surface against
normals *measured from the photo* (Meta Sapiens estimator) and snaps the outline to
the RMBG silhouette — cloth folds and facial planes come from the image, not the
generative prior. Without the model everything behaves exactly as before.

Produce the model with the one-shot local converter (quarantined dev tool — builds its
own Python venv under `tools/`, the app itself stays pure Swift/CoreML):

```bash
./tools/convert_sapiens2/convert.sh        # 0.8b default; or: convert.sh 0.4b
```

It downloads Meta's Sapiens2 normal estimator (Apr 2026 — ~46% lower angular error
than v1, license without v1's non-commercial clause), converts to fp16 CoreML straight
into `weights/SapiensNormal.mlpackage`, and validates the converted model against
torch on-device. (`docs/sapiens2_normal_coreml_colab.ipynb` does the same in
Colab/Lightning if you'd rather not have a local venv.) Then verify with:

```bash
BIN=$(swift build -c release --show-bin-path)
"$BIN/LiToSmoke" normals weights <subject_rgba.png> normals_vis.png
"$BIN/LiToSmoke" refine  weights <model_gs.ply> <subject_rgba.png> refined
```

Useful dev commands (`--selftest` works without the CoreML model — it feeds the
mesh's own normals through the full solve and should report mean disp ≈ 0.001):

```bash
LiToSmoke sculpt <weightsDir> <image> <out_base> [steps] [cfg] [seed] [bestOf] [meshRes]
LiToSmoke refine <weightsDir> <in.gs.ply> <rgba.png> <out_base> [meshRes] [iso] [grid] [--selftest]
LiToSmoke normals <weightsDir> <rgba.png> <out.png> [grid]
```

Benchmark protocol (compare every quality change against `bench/baseline/`):
`bench/run_baseline.sh [outDir]` — all of `testset/` at 25 steps, cfg 3.0, seed 7,
best-of-3, plus mesh extraction and a render per photo. NOTE: ~1.5–2.5 h per photo
on a 16 GB machine; it runs unattended (logs + summary.txt update incrementally).

Sapiens weights are CC-BY-NC-4.0 (non-commercial).

## Run it in Xcode

```bash
xcodegen generate          # regenerate LiToStudio.xcodeproj from project.yml
open LiToStudio.xcodeproj   # select the "LiToStudio" scheme and ⌘R
```

A post-build script copies `mlx.metallib` next to the app executable and symlinks
`weights/` into the app's Resources, so the in-DerivedData `.app` finds everything.

## The one non-obvious thing: `mlx.metallib`

MLX-Swift loads its GPU kernels from a file named `mlx.metallib` sitting **next to the
executable**. SwiftPM/Xcode don't emit that file, so both run paths above copy
`weights/mlx.metallib` into place. Without it you get:
`MLX error: Failed to load the default metallib`.

To point at weights stored elsewhere, set `LITO_WEIGHTS_DIR=/path/to/weights`.

## Verified working

`./run.sh engine /tmp/test.png out.ply 4` → DINOv2 → DiT (3 steps) → voxel decode →
16,570 gaussians → 240,932 colored points written to a binary `.ply`, GPU backend,
~6.5 min on a 16 GB M-series in a debug build (release is faster).
