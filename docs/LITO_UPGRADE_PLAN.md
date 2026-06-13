# LiTo upgrade plan (Phase 0 — written 2026-06-12, before any feature code)

Implementation path for the requested upgrade wave. File locations and statuses below
were verified by reading the code; details live in `LITO_ARCHITECTURE_MAP.md` and
`LITO_TECHNICAL_NOTES.md`.

## Where things live (inspection results)

- **Image upload:** `ContentView.swift` `DropZone` (drag-drop + multi-select picker) →
  `AppModel.pickImages` → `PipelineArgs.imagePaths`. Contact-sheet split:
  `LiToKit/SheetSplit.swift`, invoked in `PipelineRunner.runPipeline` phase 0.
- **Background removal:** `LiToKit/RMBG.swift` (CoreML, model absent locally) with
  Vision fallback + person-trim in `LiToKit/Preprocess.swift`.
- **Upscaling:** `LiToKit/Upscaler.swift` (Real-ESRGAN 4×, `maxDim: 4096`; model present).
- **DINOv2:** `LiToKit/Dinov2.swift`, called per view in `LiToEngine.generate`.
- **Sapiens/Sapiens2:** `LiToKit/SapiensNormal.swift` + `NormalRefine.swift` — code
  complete + self-tested, **blocked on `SapiensNormal.mlpackage`** (user's Colab).
- **SAM/SAM3:** **nowhere — zero segmentation-grounding code exists.**
- **LiTo sampling:** `LiToKit/Dit.swift` (`sample(x0:ts:conds:cfgScale:mode:…)`) driven
  by `LiToKit/LiToEngine.swift` (seed search, yaw/IoU scoring).
- **Splat output:** `LiToKit/Splat.swift` (+ `TextureProject.swift` recolor).
- **Mesh export:** `LiToKit/MeshExtract.swift` (+ marching-cubes tables).
- **Auto settings:** `LiToKit/ImageAnalyzer.swift` — **genuinely image-specific**
  (measured Sobel edge density, contrast, Laplacian sharpness, luminance, transparency;
  worst-case aggregation over views), not hard-coded. UI panel shows default→detected
  with the formulas.
- **Progress thumbnails/statuses:** generated in `PipelineRunner.runPipeline` (`.preview`
  / `.progress` / `.line` events) and `LiToEngine.generate` progress closures; rendered
  in `ContentView.swift` `StatusArea` (64 px strip) and `PreviewStrip` (80 px, viewport).
- **Viewport dot progress:** `LiToEngine` `onStepSample` (intermediate occupancy decode
  every ~steps/5) → `.cloud` event → `LiveCloudView` in `SceneKitView.swift:333`.
- **Cancellation:** belongs in three layers — UI Stop button (`GenerateButton`/
  `StatusArea`), an atomic flag checked in `runPipeline` between phases/views, and an
  `isCancelled` closure threaded into `LiToEngine.generate` + `DiT.sample`'s step loop.
  Today `AppModel.cancel()` exists but is UI-orphaned and stops nothing.
- **6-view conditioning:** `LiToEngine.generate(imageURLs:…)` (per-view DINOv2 →
  `conds: [MLXArray]`) + `MultiViewMode` in `Dit.swift`; per-view preprocessing loop in
  `runPipeline`; UI grid + mode picker in `ContentView.swift`.

## Implementable now (no missing weights/APIs) — in order

1. **D — Stop/cancel** (highest value, pure plumbing): cancel flag → `runPipeline`
   checks → engine/DiT step-loop checks → Stop button replacing Generate while running
   → temp-file cleanup + `.idle` reset. Verifiable without `lito.safetensors` only up
   to the preprocess phases; full verification after weights restore.
2. **B — Progress tree**: add structured `RunEvent.node` events at the existing per-view
   emission points; fold into a tree in `AppModel`; DisclosureGroup UI. Design in
   TECHNICAL_NOTES.
3. **C — Thumbnail expansion**: click-to-enlarge on `StatusArea`/`PreviewStrip`/drop-zone
   thumbnails (popover/sheet with label, resolution, path). Pure UI.
4. **H — Landmark list UI skeleton**: build against the `LandmarkGrounding` protocol
   with the honest `UnavailableLandmarkGrounding` empty state ("no grounding model
   installed"). No fake detections (Decision 001). Labels wait for the taxonomy file.
5. **K — Run metadata export**: write a per-run JSON (settings, seed, per-view
   yaw/IoU, timings, artifact paths) next to the PLYs — currently nothing is persisted.
6. **F — 2K policy**: decide (Decision 006), then implement — small code, needs the
   policy decision first.

## Requires missing weights / models / inputs

- **End-to-end runs, E-polish measurements, any bench:** re-download weights
  (first-run setup; `weights/` currently has only metallib + RealESRGAN) and restore
  the off-repo `testset/`. The weights-v1 GitHub release 404s — user action.
- **I — Sapiens2 diagnostics UI:** code path exists; needs `SapiensNormal.mlpackage`
  (user's Colab conversion) to show real numbers. Build the panel only when real stats
  can flow through it.
- **G — SAM3 grounding:** needs a SAM3 checkpoint + license check + a conversion route
  (CoreML or MLX) **and** the landmark taxonomy txt (not yet attached) for the label
  set. Until then: protocol + package format only (TECHNICAL_NOTES § SAM3).
- **Multi-view NormalRefine (MULTIVIEW.md M3):** Sapiens weights + weights restore.
- **K (text guidance) implementation:** any real path needs new/different weights —
  deferred (Decision 008, research doc).

## Sequencing rationale

Cancellation first because every later feature's dev loop (6-view runs are tens of
minutes) is faster once runs can be aborted; progress tree second because it's the
scaffolding the thumbnails, per-view diagnostics, and landmark statuses all hang off;
weights re-download is the gate for everything quality-related and should happen in
parallel with 1–3.
