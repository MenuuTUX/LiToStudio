# LiTo TODO — phase checklist

Markers: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked · `[?]` needs decision

## Phase A — Auto settings per image
- [x] Measured per-image analysis (Sobel edge density, contrast, Laplacian sharpness,
      luminance, transparency) → recommended steps/CFG/thresholds/seed-search
      (`Sources/LiToKit/ImageAnalyzer.swift`) — genuinely image-specific, not hard-coded
- [x] 2026-06-12 rework: old steps formula saturated at 45 for real photos. Now per-view
      metrics (subject bbox + mask ratio, in-subject detail, texture entropy, orientation
      estimate via Vision body pose, per-view 2K upscale decision) + global D-score steps
      formula with real spread (verified 42/36/32/38 on test variants). Per-view rows in
      `AnalysisPanel`; analysis runs off-main.
- [x] Run metadata JSON (`<base>_run.json`): settings, seedUsed, yaws/IoUs, artifacts,
      full analysis (PipelineRunner `RunMetadata`)
- [ ] Re-validate formula weights against bench once testset is restored (no test
      target exists in the repo, so no unit tests were added — see RUNBOOK)

## Phase B — Multi-view progress tree
- [x] 2026-06-12: per-view branch model via `EngineEvent` + `RunEvent.stage`
      (skeleton emitted up front; unavailable/skipped stages visible, never silent;
      branches persist after the run). UI: `ProgressTreeView` chips per view + trunk
      checklist with candidate progress (see TECHNICAL_NOTES § Progress tree)
- [x] CLI unaffected (`onEvent` optional; LiToSmoke keeps text output)
- [ ] Verify visually in the app on a 6-view run (built + event-flow reviewed; not yet
      observed live in the UI)

## Phase C — Thumbnail expansion
- [x] 2026-06-12: `LightboxView` — full-window, pinch zoom + drag pan + double-click
      reset, Esc/backdrop close, metadata footer; opens from tree chips, preview strip,
      and drop-zone thumbnails
- [ ] Verify visually in the app

## Phase D — Stop / cancel generation
- [x] 2026-06-12: Stop button (replaces Generate while running) with
      "finish candidate / stop now" dialog at ≥80 % or ≤5 steps remaining
- [x] Cooperative `GenCancelToken` threaded `runPipeline` → `LiToEngine.generate` →
      `DiT.sample` (per-step poll); immediate stop discards the half-sampled latent,
      finish-candidate stop decodes the best candidate as a real result
- [x] Cancelled runs clean temp files; UI shows a cancelled card + keeps the last
      intermediate cloud; in-flight tree stages flip to skipped
- [ ] Verify live: stop mid-sampling in the app (built; not yet exercised end-to-end —
      needs a real run, ~minutes per attempt)

## Phase E — Responsive viewport dot/splat progress
- [x] Basic live occupancy cloud during sampling (`onStepCloud` → `LiveCloudView`)
- [x] 2026-06-12: cadence raised (step 1, every 2nd, every step in final 30 %);
      dot size/brightness ramp with real progress; manual orbit during generation;
      turntable + shimmer toggles (both user-controllable, rotation off by default)
- [!] Decode-per-step cost NOT yet measured on a 16 GB machine — if 6-view runs swap,
      back the final-stretch cadence off to every 2 steps
- [ ] Color in the live preview: honest path = colored dots only after gaussian decode
      exists (occupancy carries no color); crossfade dots → final splat still open

## Phase F — 2K input processing
- [x] 2026-06-12: policy decided (Decision 006 — subject long side ≥ 2048 px, not
      canvas) + implemented: per-view skip/warn/upscale decision, up to two Real-ESRGAN
      passes, alpha-preserving upscale (`upscaleToMaxPreservingAlpha`), source-limited
      warnings; dims shown per stage in the progress tree
- [x] Verified via new `LiToSmoke upscale` subcommand (300²→4096² in 2 passes; RGBA
      icon keeps alpha)
- [ ] Verify in-app on a real 6-view set (needs a live run)

## Phase G — SAM3 landmark grounding
- [x] 2026-06-12: taxonomy attached → `docs/LANDMARK_TAXONOMY.txt`; core set + § L
      priors embedded in `LiToKit/LandmarkGrounding.swift`; `LandmarkPackage` JSON
      exported per run with cross-view visibility matrix
- [x] 2026-06-12 (deep): REAL worker built — `tools/backend/sam3_worker.py`
      (facebook/sam3 via transformers, text-prompted, per-token masks/boxes/conf,
      honest not_detected/failed) + `Sam3Backend` Swift adapter + pipeline wiring
      (runs pre-engine, matrix flips to real outcomes, masks → `<base>_masks/`,
      UI shows detections + mask thumbnails)
- [x] 2026-06-12 (later): **gate bypassed — native CoreML backend working.**
      `AllanVester/SAM3.1-CoreML-FP16` downloaded to `weights/sam3-coreml/`;
      `Sam3CoreML.swift` driver (CLIP-BPE prompt tokens baked, mask-derived boxes,
      CPU+GPU); preferred over the worker in the pipeline; verified visually on
      testset photos + batch via `LiToSmoke ground` (Decision 012)
- [x] 2026-06-12 (finalized — Decision 013): person-mask gating kills the whole-person
      fallback + background speckle; presence floor 0.51 gives real cross-view
      discrimination (back: no face/navel/chain; front: face/navel/charms); detections
      shown as **region-over-photo overlays** (not raw masks); text field → live SAM 3.1
      concept (`userConcept` in the package). Verified `sam3` / `sam3concept` / `ground`
- [!] facebook/sam3 (official, gated) still pending Meta approval — wanted as a
      PARITY CHECK for the community conversion, no longer a blocker
- [ ] Consumption by the DiT still needs a conditioning fine-tune
      (`DiT.sample(conds:)` is the seam)
- [ ] Sheet-split runs skip grounding (package/view-count mismatch — logged)

## Phase H — Semantic object/landmark list UI
- [x] 2026-06-12: `Views/LandmarkPanelView.swift` — taxonomy tokens grouped by
      category, cross-view visibility matrix (● detected ○ expected – not ? unknown),
      per-token inspection, per-view label correction menus, honest no-backend banner,
      human-feature backend diagnostics footer
- [ ] Verify visually in the app (built; not yet observed live)

## Phase I — Sapiens2 diagnostics / pose backend
- [x] Refinement pipeline code + self-tests (`NormalRefine.swift`, `SapiensNormal.swift`);
      `[refine]` log lines already emit Σ|corr|, moved verts, displacement stats
- [x] 2026-06-12: framing classifier + raised-hand via Apple Vision (labeled as Vision)
- [x] 2026-06-12 (deep): REAL Sapiens2 pose backend — facebook/sapiens2-pose-0.4b
      installed + verified (`LiToSmoke ground`: 308 Goliath keypoints, face/hands/feet
      group summaries, per-view records in package + metadata + UI; subject-box crops
      instead of RTMDet — Decision 011)
- [!] Mesh *normal* refinement still blocked on `weights/SapiensNormal.mlpackage`
      (the user's Colab conversion — separate artifact from the pose model)
- [?] Opportunity discovered 2026-06-12: `facebook/sapiens2-normal-0.8b` is ALREADY
      in the local HF cache (3.3 GB) — a `sapiens2_normal_worker.py` could feed
      `NormalRefine` real normals via the Python backend, bypassing the Colab CoreML
      conversion entirely. Decide: worker vs CoreML path
- [ ] Surface refine stats in the UI once the normal model exists
- [ ] Optional: larger sapiens2-pose-1b as a quality option (documented in backend README)

## Phase M — RMBG-2.0 background removal (deep stage, 2026-06-12)
- [x] License accepted + weights downloaded (user's HF account); Python worker
      (`rmbg_worker.py`) produces real full-res RGBA cutouts — verified standalone and
      through the Swift adapter (`LiToSmoke ground`)
- [x] Pipeline: CoreML path when `weights/RMBG2.mlpackage` exists, else worker batch;
      cutouts persisted as `<base>_v{i}_cutout.png`; texture skip now warns with exact
      install steps; statuses in run metadata
- [!] CoreML conversion permanently blocked at coremltools ≤ 9.x
      (`torchvision::deform_conv2d`) — `convert_rmbg2.py` kept for the future
- [ ] Verify photo texture end-to-end on a 6-view run with worker cutouts

## Phase J — Splat + mesh output
- [x] 3DGS splat PLY (SH3), colored point cloud, marching-cubes mesh PLY/OBJ
- [x] HD photo-texture backprojection onto splat + mesh (≥2 views; self-tested synthetic)
- [x] 2026-06-11/12: first end-to-end 6-view runs in the app (user-verified — splat +
      178k-vert mesh produced; bench comparison still open)
- [x] 2026-06-12: Export menu (splat / mesh .ply+.obj / run metadata / landmark
      package) + mesh-quality caption (marching cubes from 64³ latent, no hole filling
      beyond component cleanup)
- [x] 2026-06-12 (deep): mesh component cleanup — islands < 2 % of the largest shell
      removed (union-find + compaction in `MeshExtract`); hole filling beyond that
      intentionally NOT implemented (would be cosmetic invention at 64³ — see
      TECHNICAL_NOTES § Mesh)
- [ ] Real surface reconstruction / decimation (future: MeshDecoder,
      displacement baking); GLB export
- [ ] Bench validation of multi-view vs single (testset exists at ~/Downloads/testset)

## Phase K — Optional user text prompt guidance
- [x] Research: current checkpoint has **no** text conditioning path
      (`docs/LITO_PROMPT_GUIDANCE_RESEARCH.md`)
- [x] 2026-06-12: optional guidance field added, clearly labeled as NOT consumed by
      the pipeline; recorded in run metadata (`userPrompt`) only
- [?] Decide whether to pursue any of the researched options (recommendation: defer)

## Phase L — Documentation / verification
- [x] Context-management system (CLAUDE.md + docs/LITO_*.md) — 2026-06-12
- [ ] Re-download weights (first-run setup) → first end-to-end multi-view run
- [ ] Restore testset + run `bench/run_baseline.sh` multi-view comparison
- [!] Fix weights-v1 GitHub release (404; user must `gh release create`)
- [ ] Keep CONTEXT_PACK "last worked on" current every session
