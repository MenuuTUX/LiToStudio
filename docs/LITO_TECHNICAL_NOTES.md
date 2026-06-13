# LiTo technical notes

Implementation details per feature area. Sections marked *design* describe intended
work, not existing code — the architecture map carries the authoritative status.

## Auto-settings formula (implemented — ImageAnalyzer.swift, reworked 2026-06-12)
The old `steps = 20 + 80·ε` saturated at the 45 clamp for nearly every real photo
(frame-wide edge density ε ≈ 0.4+) — every upload landed on 45. The rework measures
per view, then aggregates.

Per view (EXIF-upright, downsampled): dimensions; pre-masked flag; ℓ luminance;
σ contrast (sd/80); λ sharpness (256² Laplacian variance / 300); ε edge density at 128²
for the whole frame AND inside the estimated subject box; τ texture (normalized 32-bin
gray-histogram entropy inside the box); subject box + mask-area ratio + crop ratio
(alpha bbox when pre-masked, else border-color-distance heuristic — marked *unreliable*
and skipped when the border σ > 45, i.e. busy background); orientation estimate
(front / oblique L–R / profile L–R / back / unknown) from Vision body-pose landmark
visibility (nose offset from shoulder midline splits front vs oblique; single ear/eye
side ⇒ profile; shoulders without face points ⇒ back; non-person ⇒ unknown); per-view
2K upscale decision (see § 2K policy).

Global aggregation: detail = mean(min(1, ε_subj/0.40)), τ = mean, res =
mean(min(1, subjectLongSide/2048)); λmin/σmin/ℓmin = minima ("weakest view needs help").
- **steps** = clamp(16 + 26·D + min(8, 2·(V−1)) − 6·blur, 14, 48) rounded to 2, where
  D = 0.45·detail + 0.30·τ + 0.25·res and blur = clamp(1 − λmin/0.35, 0, 1).
  More measured detail/texture/resolution and more views → more steps; soft inputs →
  fewer (extra steps sharpen noise, not shape). Means (not maxima) keep one busy view
  from saturating the formula.
- cfg = min(3 + 2·max(0, 0.25 − σmin), 3.5) · seeds = best-of-3 single / best-of-2 multi
- useRMBG = not all premasked · useUpscaler = any view's per-view 2K decision
- occupancy 0.25 iff detail > 0.75 ∧ σmin < 0.30 · opacity 0.12 clean&simple else 0.08
Verified spread (LiToSmoke analyze, same wallpaper at three qualities): 2400px sharp →
42 · 300px → 36 · 900px blurry → 32 · all three as views → 38.
Every decision carries a `Note` with the math; per-view rows + global notes render in
`AnalysisPanel`; the whole analysis is persisted in the run metadata JSON
(`<base>_run.json`, see § Run metadata).

## Run metadata (implemented 2026-06-12)
`RunMetadata` (PipelineRunner.swift) writes `<base>_run.json` next to the artifacts:
schema version, ISO date, input paths, all settings (steps/cfg/mode/thresholds/flags),
`seedUsed` (reproduces the kept candidate), per-view yaw/IoU estimates, artifact file
names, and the full auto-detect analysis when auto was on.

## Progress tree (implemented 2026-06-12)
- Engine emits typed `EngineEvent`s (`Sources/LiToKit/EngineEvents.swift`:
  viewPreprocessing/viewEncoding/viewEncoded/samplingStep/candidateDone/decoding/
  decodedGaussians/writingOutput) via a new `onEvent` closure on `LiToEngine.generate`.
- `runPipeline` emits `RunEvent.stage(StageUpdate)` (view branch or trunk) plus
  `RunEvent.sampling(...)` for the Stop logic. A skeleton of ALL stages is emitted up
  front; absent models are `unavailable`, disabled/not-run work is `skipped` with the
  reason — never silently done. Per-view branch order matches the real pipeline:
  original → upscale → background → crop → DINOv2 → Sapiens (unavailable/skipped) →
  SAM3 (unavailable) → view token. Trunk: conditioning merge → candidate 1..N →
  best candidate → gaussian decode → photo texture → mesh → result.
- `AppModel` folds updates into `viewStages: [[StageRecord]]` + `trunkStages`
  (upsert: status/label always win, thumbnail/dims/detail only overwrite when carried;
  records persist after the run; failures mark in-flight stages failed).
- UI: `Views/ProgressTreeView.swift` — chip row per view (38 px thumbs, status-colored
  borders, tooltips with detail/dims/timestamp), trunk checklist with per-candidate
  progress bar. Any thumbnail (tree chips, preview strip, drop zone) opens
  `LightboxView`: full-window, pinch zoom (1–8×), drag pan, double-click reset,
  Esc/backdrop/button close, metadata footer (view index, stage, dims, status).

## Cancellation (implemented 2026-06-12)
- `GenCancelToken` (`Sources/LiToKit/EngineEvents.swift`): NSLock-guarded mode flag,
  escalation-only (`none → afterCandidate → immediate`). Polled by the pipeline thread
  between phases/views, by the engine per candidate, and by `DiT.sample` before every
  Heun step (`shouldStop` closure).
- **Immediate**: the step loop exits, the half-integrated latent is *discarded*
  (never decoded — it would look like a finished result), the engine throws
  `EngineError.cancelled`, `runPipeline` yields `.cancelled`, temp files are removed,
  the UI shows "Stopped — no result produced" and keeps the last intermediate
  occupancy cloud visible in the viewport. In-flight stage chips flip to skipped
  ("stopped by user").
- **Finish candidate**: the candidate completes sampling + scoring, the seed loop
  breaks, the best candidate so far is decoded and written as a real splat/point-cloud
  result; photo texture and mesh extraction are skipped (marked so in the tree);
  phase ends `.done` with the viewer showing the kept splat.
- Stop button replaces Generate while running. When the current candidate is ≥ 80 %
  done or ≤ 5 steps remain, a dialog offers "Finish candidate, then stop" /
  "Stop immediately" / "Keep generating"; otherwise Stop is immediate.
- MLX caveat: an in-flight `eval()` can't be interrupted — stop latency is ≈ one
  sampling step or one decode; the button shows "Stopping…" meanwhile.
- Cancel during preprocessing (before any candidate) is always immediate.

## Viewport progressive rendering (implemented 2026-06-12)
- Cadence: `LiToEngine` `onStepSample` decodes the predicted-final latent's occupancy
  at step 1, then every 2nd step, then **every** step in the final 30 % (each emission
  costs one voxel+occupancy decode; raised from every ~steps/5 — cost not yet measured
  on a 16 GB machine, watch for swap on 6-view runs). Early decodes may legitimately
  be empty/sparse — the dots genuinely grow in; nothing is synthesized.
- `RunEvent.cloud` carries step/total → `AppModel.liveCloudProgress` →
  `LiveCloudView(progress:autoRotate:shimmer:)`: dot size 2→3.5 px and brightness/
  opacity ramp with real sampling progress (neutral shading — occupancy decodes carry
  no color; real color only exists after the final gaussian decode, so none is shown
  before then).
- Manual orbit always on (`allowsCameraControl`, orbitTurntable). Turntable is
  toggleable and **off by default** (Decision 005 amended). Subtle "building" shimmer
  (±1.2 % scale pulse) toggleable via settings + viewport button, off automatically in
  the cancelled-state preview.

## 2K image policy (decided + implemented 2026-06-12 — Decision 006)
- **"2K" = estimated subject (foreground bbox) long side ≥ 2048 px** (Option B).
  Conditioning crops to the subject before the 518² downsample, and texture
  backprojection samples subject pixels — canvas/background never reaches either
  consumer, so a canvas threshold (Option A) rewards the wrong thing. Fallback:
  canvas long side when no reliable subject estimate exists (busy background).
- Analyzer and pipeline share the estimate (`ImageAnalyzer.subjectBoxEstimate`) and
  constants (`subjectTargetPx` = 2048, `canvasCapPx` = 4096), so they can't disagree:
  - subject ≥ 2048 and sharp (λ ≥ 0.20) → skip, "no upscale needed".
  - canvas ≥ 4096 but subject < 2048 → skip + **warning** "consider a tighter crop"
    (more upscaling can't help inside the memory cap).
  - else Real-ESRGAN 4×; if the subject is still < 2048 and the canvas has headroom,
    exactly ONE more pass (cascaded super-resolution — real model passes, never plain
    resampling); if still short, a source-limited warning lands in the log and the
    stage detail. Canvas hard cap 4096 px (memory budget, 6 × RGBA temporaries).
- Alpha: `Upscaler.upscaleToMaxPreservingAlpha` — the model only sees RGB, so the
  alpha plane is resampled separately (high-quality CG interpolation) and re-attached
  premultiplied. Before this fix a pre-masked RGBA cutout lost its mask in the
  upscaler (transparent → black).
- Each pass resamples once; no repeated resizing. Per-view dims are surfaced in the
  progress tree (original → upscaled → 518² conditioning crop) and analyzer rows.
- Verified: `LiToSmoke upscale weights/RealESRGAN_x4.mlmodel <img>` — 300² opaque →
  4096² in 2 passes; 1024² RGBA icon → 4096², `hasAlpha: yes`.

## SAM3 finalization (2026-06-12 — Decision 013)
Three issues from the first integration, all fixed:
- **Whole-person fallback** (the "face" mask covering the entire body): the detector
  returns its dominant-object query at the score floor when a concept is absent.
  Fix: `ground`/`detect` take the **person silhouette** (288², from the RMBG cutout —
  `personMask288`); every query mask is intersected with it (kills background
  speckle), and detection walks the score-ranked queries skipping any whose mask
  fills > 70 % of the person (`maxPersonCoverage`) — that's the fallback, not a part.
  If no plausible part survives, the result is an honest `not_detected` (we never
  return the whole-person mask). `personCoverage` is recorded + shown.
- **Presence threshold**: scores cluster at sigmoid(0) ≈ 0.500 for absent concepts and
  rise for present ones; the gap measured across real multi-view photos is kept ≥ 0.514
  / rejected ≤ 0.506, so the floor is **0.51** (`presenceFloor`, override
  `LITO_SAM3_THRESHOLD`). Verified: back view → face/navel/chain/charms correctly
  `not_detected`, hair/gloves/cargo/pockets/nails detected; front view → face/navel/
  charms detected. Real cross-view discrimination.
- **Display**: the UI showed the raw binary mask (meaningless alone). Now
  `writeOverlay` renders the region over the original photo — kept colour + green
  tint inside, dimmed to 32 % outside, green bbox — composited in one top-down RGBA
  buffer (an earlier two-context version flipped the image upside-down). Stored as
  `<base>_masks/v{i}_{id}_overlay.png`; `LandmarkObservation.overlayPath`; the panel
  thumbnails + lightbox use the overlay, the raw mask stays as the data artifact.

## Text concept guidance (2026-06-12 — Decision 013)
The optional text field, when SAM 3.1 is installed, is segmented as a live concept in
every view: `ClipTokenizer` (`tokenize_clip.py`, CLIP-BPE, needs only the venv) →
`Sam3CoreML.groundConcept` (same gating/overlay as the core set, id "USER") →
`LandmarkPackage.userConcept { phrase, perView: [observation?] }` → shown in the
Semantic Landmarks panel (region overlays per view) + exported in the package.
Verified: "leather glove" → the glove, "cargo pants" → the lower body, on the back
view. **It does NOT alter generated geometry** — the LiTo checkpoint has no text
pathway (Decision 008 unchanged); it is real text→region grounding, recorded and
shown, and the field copy + panel state that explicitly. Without SAM 3.1 the field is
metadata-only as before.

## Model backends — Python workers (installed 2026-06-12)
`tools/backend/` (see its README for exact install/auth steps): one uv venv
(python 3.12, torch 2.12, transformers 5.12, MPS). Swift side:
`Sources/LiToKit/PythonBackend.swift` — worker discovery (`LITO_BACKEND_DIR` or repo
walk), availability = venv + HF-cache presence, one process per batch, manifest JSON
in / one JSON out / artifacts as files, 1 s cancel polling (immediate stop terminates
the worker), stdout/stderr via temp files (no pipe deadlocks). Workers run in the
pipeline *between* preprocessing and engine load — they exit before MLX allocates.
Status at install time:
- **RMBG-2.0** (`rmbg_worker.py`): WORKING — license accepted on the user's HF
  account, weights cached; real full-res RGBA cutouts verified via `LiToSmoke ground`.
  CoreML conversion blocked (deform_conv2d — Decision 011).
- **Sapiens2 pose 0.4b** (`sapiens2_worker.py`): WORKING — ungated, cached; real
  Goliath-308 keypoints verified (testset photo: face 130/136 @0.97, hands 32/32
  @0.95, feet honestly 0/8 on an above-knee crop).
- **SAM3: WORKING natively** — `Sources/LiToKit/Sam3CoreML.swift` drives the
  community CoreML conversion (`weights/sam3-coreml/`, Decision 012): per image, one
  ImageEncoder pass (1008² → FPN pyramid) then per-prompt TextEncoder (cached) +
  Detector (200 queries → 288² mask logits + presence scores). Finalized
  2026-06-12 (Decision 013): see § SAM3 finalization below.
  The gated transformers worker (`sam3_worker.py`) remains the fallback / parity
  check once facebook/sam3 access is approved.

## SAM3 landmark grounding (real worker 2026-06-12 — weights gated)
- Taxonomy attached and canonical at `docs/LANDMARK_TAXONOMY.txt`; the core set
  (L001–L012) + per-view visibility priors (§ L) are embedded in
  `Sources/LiToKit/LandmarkGrounding.swift` so code and UI can't drift from it.
- Backend: `Sam3Backend` (PythonBackend.swift) runs `sam3_worker.py` — text-prompted
  concept segmentation per core-set token (prompt phrases live on `LandmarkToken`),
  per-view boxes (normalized xywh) + mask PNGs (`<base>_masks/v{i}_{id}.png`) +
  confidences, with honest `not_detected` / `failed` statuses per token.
  `UnavailableLandmarkSegmenter` remains only as the no-backend placeholder.
  Weights are gated (Meta approval) — until granted, availability is false and the
  matrix stays priors-only. Nothing fabricates observations.
- View labels (`ViewLabel`): front / front_right_oblique / right_profile / back /
  front_left_oblique / left_profile / unknown. Inference precedence:
  **user override > filename token > Vision pose estimate > unknown** — correctable
  per view in the Semantic Landmarks panel (`AppModel.setViewLabel`).
- `LandmarkPackage` (exported as `<base>_landmarks.json` per run, referenced from run
  metadata): per-view entries x_i = {imagePath, viewLabel+source, poseFeatures
  (Apple Vision: framing/orientation/raised-hand/suggested focus), expectedTokens
  (taxonomy priors), observations (empty — no backend)}, plus the cross-view
  visibility matrix (rows = core tokens, cells = `detected:<conf>` | `expected` |
  `not_expected` | `unknown`). `consumedByGenerator` is **false** with the reason in
  `consumptionNote`: the DiT cross-attends DINOv2 image tokens only.
- Integration point when a backend + fine-tune exist: `DiT.sample(conds:)` already
  accepts arbitrary token streams; a projected landmark stream joins there.
- Verified: `LiToSmoke landmarks <img,…> [out.json]` — filename labels
  (right_profile/back), pose-estimate labels on person photos, priors matching § L,
  zero detections, JSON written.
- Sheet-split caveat: the package is built from the *dropped* files; a contact sheet
  exports one entry for the sheet, not per split view (future work).

## Mesh output + exports (2026-06-12)
- Splat/Mesh/Points viewers already existed; added an **Export menu** in the viewer
  bottom bar: splat .ply, mesh .ply/.obj, run metadata .json, landmark package .json —
  save-panel copies of the real artifacts, items only shown when the file exists.
- Pipeline reality: 256³ density grid from the splats (geometry information still
  bounded by the 64³ occupancy scaffold) → marching cubes → Taubin smoothing (λ/µ,
  15 iter) → **component cleanup (new): islands < 2 % of the largest shell removed,
  union-find + vertex compaction (`MeshExtract.removeSmallComponents`)** → nearest-
  splat vertex colors (+ photo backprojection when ≥ 2 cutout views).
- Marching cubes over the closed density field is watertight except where the field
  is genuinely absent — "hole filling" beyond component cleanup would be cosmetic
  invention at this resolution and is intentionally NOT implemented.
- True high-definition mesh requires: a higher-resolution geometry field (MeshDecoder
  on the sculpt roadmap), measured-normal displacement (Sapiens refinement once its
  CoreML normal model exists), UV texture baking from the 4K views, and optionally
  external remeshing/retopology. GLB export not supported (PLY/OBJ only).

## Sapiens2 / human-feature guidance (REAL pose backend since 2026-06-12)
- `SapiensPoseBackend` + `sapiens2_worker.py`: facebook/sapiens2-pose-0.4b
  (transformers, MPS), top-down with the app's subject-box as the person crop
  (Decision 011). Per view: 308 keypoints (Goliath names), group summaries
  (body/face/hands/feet visible counts + mean confidence), raised-hand cue, full
  keypoints in `<base>_pose/v{i}_pose.json`. Lands in `ViewEntry.sapiensPose`, the
  run metadata, the panel's view rows, and the per-view "Sapiens2 pose" tree chips.
  The CoreML *normal* model for mesh refinement is a separate artifact and remains
  blocked (Colab conversion).

## Apple Vision pose features (pre-analysis; distinct from Sapiens2)
- What runs TODAY (real, Apple Vision body pose — explicitly labeled as such, never
  presented as Sapiens output): per-view orientation estimate, **framing classifier**
  (face_crop / upper_body / torso / head_to_knees / full_body from the lowest
  confidently-visible joint band), raised-hand detection (wrist above nose height).
  Exposed in `ViewAnalysis` (`framing`, `raisedHand`), the analyzer UI rows, and as
  `PoseFeatures` in the landmark package — including a `suggestedFocus` string per
  framing (face crop → facial structure/identity detail; upper body → head/shoulder/
  arm/hand structure; head-to-knees → body pose incl. hips/knees; back/profile →
  rear/side hair + garment structure). That is the dispatch plan for Sapiens2 once
  installed; today it is a recommendation only.
- The Semantic Landmarks panel shows backend status honestly: "Apple Vision body
  pose — active" / "Sapiens2 — not installed".
- Sapiens itself: `SapiensNormal.swift` + `NormalRefine.refine` (stats: Σ|corr|,
  moved verts, displacement, abort reason via `[refine]` lines) remain blocked on
  `SapiensNormal.mlpackage` from the user's Colab
  (`docs/sapiens2_normal_coreml_colab.ipynb`; sapiens2 0.8b). How it guides LiTo when
  present: post-hoc mesh re-sculpting against measured normals (view 1 today,
  multi-yaw = MULTIVIEW.md M3) — it does not condition the DiT.

## Mesh conversion notes (implemented)
- Marching cubes over the gaussian density field (`MeshExtract.swift`), vertex colors,
  PLY + OBJ. z-up coordinates throughout; SceneKit viewers rotate −π/2 about x.
- Density-field threshold interacts with `occupancyThreshold`/`opacityThreshold` —
  re-tune only via bench (Decision 007). Future: UV texture + displacement baking
  (sculpt roadmap), TripoSG comparison.

## Optional text prompt guidance (researched — Phase K)
See `LITO_PROMPT_GUIDANCE_RESEARCH.md`. Summary: no text pathway exists in the
checkpoint; all options require new weights or a different checkpoint; deferred
(Decision 008).
