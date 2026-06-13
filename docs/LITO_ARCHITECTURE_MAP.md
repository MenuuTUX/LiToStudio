# LiTo architecture map

Where every system lives, what it actually does today, and its honest status.
Statuses: **implemented** · **partial** · **placeholder** · **missing** · **blocked**.
Verified against the code 2026-06-12.

Targets (Package.swift): `LiToStudio` (SwiftUI app) · `LiToKit` (MLX engine) ·
`LiToConvertCore`/`LiToConvert` (ckpt→safetensors) · `LiToSmoke` (dev CLI harness).

## Upload / image input — implemented
- `Sources/LiToStudio/Views/ContentView.swift` (`DropZone`) — drag-drop or file picker,
  multi-select; >1 image renders a thumbnail grid ("N views — multi-view conditioning").
- `Sources/LiToStudio/Core/AppModel.swift` — `inputImageURLs: [URL]`, `pickImages`
  resets state and triggers auto-analysis.
- `Sources/LiToKit/SheetSplit.swift` — one image that is a contact sheet (OCR text
  erase → foreground mask → connected components) → N per-view crops. Self-tested.

## Background removal — implemented (REAL via Python worker since 2026-06-12)
- Active path: `RMBGWorkerBackend` (`Sources/LiToKit/PythonBackend.swift`) →
  `tools/backend/rmbg_worker.py` (briaai/RMBG-2.0, PyTorch/MPS, license accepted on
  the user's HF account) — full-res RGBA cutouts, persisted as
  `<base>_v{i}_cutout.png` run artifacts. Verified via `LiToSmoke ground`.
- `Sources/LiToKit/RMBG.swift` — CoreML driver, used only if `weights/RMBG2.mlpackage`
  ever exists; conversion currently impossible (deform_conv2d — Decision 011),
  `tools/backend/convert_rmbg2.py` kept for future coremltools.
- `Sources/LiToKit/Preprocess.swift` — Apple Vision fallback
  (`VNGeneratePersonSegmentationRequest`), person-mask trim, low-light normalize,
  EXIF-upright loading, cond-crop to 518² RGBA.

## Upscaling / resizing — implemented (2K policy 2026-06-12)
- `Sources/LiToKit/Upscaler.swift` — Real-ESRGAN 4× CoreML, `upscaleToMax(maxDim:)` +
  `upscaleToMaxPreservingAlpha` (alpha plane resampled separately, re-attached
  premultiplied). Pipeline applies Decision 006: subject ≥ 2048 px target, skip/warn/
  cascade-up-to-2-passes, 4096 px canvas cap. Self-test: `LiToSmoke upscale`.

## DINOv2 conditioning — implemented
- `Sources/LiToKit/Dinov2.swift` — encodes 518² RGBA → 1374×2048 tokens, one per view.
- No camera/pose embedding — conditioning is pose-free, which is what makes sampling-time
  multi-view possible.

## LiTo sampling — implemented
- `Sources/LiToKit/Dit.swift` — flow-matching DiT, Heun ODE, CFG (uncond via
  y_embedding; the blobby-output bug was missing CFG — keep cfg=3).
  `MultiViewMode`: `multidiffusion` (avg cond velocities, single uncond eval) ·
  `stochastic` (round-robin) · `concat` (token concat, off-distribution).
- `Sources/LiToKit/LiToEngine.swift` — orchestrates preprocess → DINOv2 → sample →
  decode → write; best-of-N seed search scored by silhouette IoU (mean over views,
  per-view 5° yaw sweep when N>1); emits `GenResult(viewYaws, viewIoUs)`.
- `Sources/LiToKit/VoxelDecoder.swift`, `Trellis.swift` — latent → 64³ occupancy.
- `Sources/LiToKit/GaussianDecoder.swift`, `Splat.swift` — gaussians (SH3) → PLY.
- Weights loading: `Weights.swift` (fp16 default, `LITO_FP32=1` to force fp32).

## Progress UI — implemented (2026-06-12)
- Event channel: `RunEvent` in `Sources/LiToStudio/Core/PipelineRunner.swift`
  (`progress`, `line`, `preview`, `cloud`, `result`, `failed`, `cancelled`,
  `stage(StageUpdate)`, `sampling`). Typed engine events:
  `Sources/LiToKit/EngineEvents.swift` (`EngineEvent`, `GenCancelToken`).
- `Sources/LiToStudio/Views/ProgressTreeView.swift` — per-view chip branches + trunk
  checklist + `LightboxView` (zoom/pan/Esc). State folded into
  `AppModel.viewStages`/`trunkStages` (persists after the run). Old preview strip kept
  as viewport overlay + fallback.

## Viewport renderer — implemented
- `Sources/LiToStudio/Views/SplatView.swift` — Metal 3DGS via MetalSplatter (needs its
  compiled default.metallib; run.sh handles plain swift-build).
- `Sources/LiToStudio/Views/SceneKitView.swift` — mesh/point-cloud viewer +
  `LiveCloudView` (line 333): intermediate occupancy dots while sampling, fed by
  `LiToEngine` `onStepCloud` every ~steps/5 steps. Basic but working (Phase E polish).

## Cancellation / job state — implemented (2026-06-12)
- `GenCancelToken` (`Sources/LiToKit/EngineEvents.swift`) polled by `runPipeline`,
  `LiToEngine.generate`, and `DiT.sample` (per Heun step). Two modes: immediate
  (discard, `.cancelled` phase, temp cleanup, last dot cloud preserved) and
  finish-candidate (best candidate decoded as a real result, texture/mesh skipped).
  Stop button + near-completion dialog in `ContentView.swift` `GenerateButton`.
  Details: TECHNICAL_NOTES § Cancellation. Live in-app verification still pending.

## SAM3 / segmentation — REAL, native CoreML (2026-06-12)
- **Active backend:** `Sources/LiToKit/Sam3CoreML.swift` + `weights/sam3-coreml/`
  (AllanVester/SAM3.1-CoreML-FP16: ImageEncoder/TextEncoder/Detector, fp16,
  ungated, SAM License). Pre-tokenized CLIP-BPE prompts in `prompt_tokens.json`
  (`tools/backend/make_sam3_tokens.py`). Selected first in the pipeline
  (`Config.sam3CoreMLDir`); models load CPU+GPU (Detector can't compile for ANE) and
  are scoped to release before MLX loads. Details + caveats: Decisions 012–013.
- **Finalized (Decision 013):** person-mask gating (`personMask288` from the RMBG
  cutout) rejects the whole-person fallback + strips speckle; presence floor 0.51;
  detections rendered as region-over-photo overlays (`writeOverlay` →
  `LandmarkObservation.overlayPath`); the UI (`LandmarkPanelView`) shows overlays +
  the `userConcept` row. Text concept: `ClipTokenizer` (`tokenize_clip.py`) +
  `Sam3CoreML.groundConcept`. Verified: `LiToSmoke sam3 / sam3concept / ground`.
- `Sources/LiToKit/LandmarkGrounding.swift` — `ViewLabel` inference, taxonomy core
  set (+ model prompt phrases) + § L priors, `LandmarkPackage` with
  detected/not_detected/failed matrix cells via `applying(sam3:pose:)`.
- `Sam3Backend` (PythonBackend.swift) → `tools/backend/sam3_worker.py`
  (facebook/sam3 via transformers, text-prompted concept segmentation; masks to
  `<base>_masks/`). **Checkpoint gated pending Meta approval** — adapter activates on
  `hf download facebook/sam3`; until then the matrix is priors-only and the UI says
  so. Self-tests: `LiToSmoke landmarks` (priors), `LiToSmoke ground` (real backends).
- UI: `Sources/LiToStudio/Views/LandmarkPanelView.swift` — landmark list + matrix
  (● detected ◌ not-detected ○ expected – not ✕ failed ? unknown), mask thumbnails →
  lightbox, view-label correction, real backend diagnostics (Phase H).

## Sapiens2 / human pose features — REAL pose backend active (2026-06-12)
- `SapiensPoseBackend` (PythonBackend.swift) → `tools/backend/sapiens2_worker.py`:
  facebook/sapiens2-pose-0.4b (ungated, cached), Goliath-308 keypoints
  (name table: `tools/backend/goliath_keypoints.json`), subject-box person crops
  (Decision 011), per-view group summaries + raised-hand into
  `ViewEntry.sapiensPose`, run metadata, panel rows, and tree chips. Verified on
  testset photos via `LiToSmoke ground`.
- Apple Vision body pose remains the pre-analysis layer (orientation/framing at image
  pick time) and is always labeled as Vision, never as Sapiens output.
- `Sources/LiToKit/SapiensNormal.swift` — CoreML wrapper, locates
  `SapiensNormal.mlpackage`/`.mlmodelc` in weights dir.
- `Sources/LiToKit/NormalRefine.swift` — normals → screened-Poisson depth → sculpt
  camera-facing surface + silhouette snap; emits `[refine]` diagnostics. Self-tested.
- Blocked on the user's Colab conversion (`docs/sapiens2_normal_coreml_colab.ipynb`);
  `tools/convert_sapiens2/convert.sh` referenced by README does **not** exist locally.
- Multi-yaw refinement (per-view sculpt) not yet implemented (MULTIVIEW.md M3).

## Splat output — implemented
- `Sources/LiToKit/Splat.swift` — `_gs.ply` (full 3DGS, z-up) + `_pc.ply` preview.
- `Sources/LiToKit/TextureProject.swift` — backprojects photo pixels onto splat/mesh
  using per-view yaw + IoU weights (≥2 views). Self-tested on synthetic two-view sphere.

## Mesh export — implemented
- `Sources/LiToKit/MeshExtract.swift` (+ `MarchingCubesTables.swift`) — 256³ density
  grid → marching cubes → Taubin smoothing → component cleanup (islands < 2 % of the
  largest shell, added 2026-06-12) → vertex colors → `_mesh.ply`/`_mesh.obj`.
  Detail bounded by the 64³ latent; honest limits in TECHNICAL_NOTES § Mesh.

## Python model backend — new (2026-06-12)
- `tools/backend/` — uv venv + `rmbg_worker.py` / `sam3_worker.py` /
  `sapiens2_worker.py` / `convert_rmbg2.py` + `goliath_keypoints.json` + README
  (install/auth per model). Swift bridge: `Sources/LiToKit/PythonBackend.swift`
  (process-per-batch, JSON contract, cancel-aware, temp-file IO). Workers run between
  preprocessing and engine load in `runPipeline` — never concurrent with MLX.

## Run metadata — implemented (2026-06-12)
- `RunMetadata` (PipelineRunner.swift) writes `<base>_run.json` per completed run:
  settings, seedUsed, per-view yaw/IoU, artifact names, full auto-detect analysis,
  landmark-package file reference, and the optional `userPrompt` (recorded, never
  consumed). `<base>_landmarks.json` is written alongside. Export menu in the viewer
  copies splat / mesh (.ply/.obj) / both JSONs anywhere via save panel.
  Output dir: `~/Library/Application Support/LiToStudio/results/` (`Config.outputDir`).

## Setup / weights install — implemented
- `Sources/LiToStudio/Core/WeightsInstaller.swift` + `Views/SetupView.swift` — first-run
  download (checksummed, resumable) + local ckpt conversion via `LiToConvertCore`.
- `Config` (PipelineRunner.swift) — weights dir resolution, `LITO_WEIGHTS_DIR` override,
  metallib colocation. **Current local state (2026-06-12): full engine weights live in
  `~/Library/Application Support/LiToStudio/weights/` (lito.safetensors + tokenizer);
  the repo `weights/` has only metallib + RealESRGAN. RMBG2 and SapiensNormal
  .mlpackages are still absent everywhere.** Note: `./run.sh` exports
  `LITO_WEIGHTS_DIR=$PWD/weights`, which would *bypass* the App Support weights —
  launch the app binary directly (or copy weights) for engine runs. The GitHub
  weights-v1 release still 404s (user must recreate it).

## Dev harness / bench — implemented (inputs absent)
- `Sources/LiToSmoke/main.swift` — `smoke|dino|dit|voxel|gs|rmbg|cond|engine|sculpt|
  split|texture|mesh|refine|normals|render|score|gscheck|coreml` subcommands.
- `bench/run_baseline.sh` + `bench/EXPERIMENTS.md` — quality protocol (25 steps/cfg 3/
  seed 7/best-of-3); needs `testset/` (off-repo) and full weights.
- CI: `.github/workflows/swift.yml` — `swift build` + `swift test` on macos-latest
  (no test target exists, so the test step is a no-op).
