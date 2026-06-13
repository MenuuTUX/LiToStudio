# LiTo context pack — read this first

One-page project memory. Update the "Last worked on / resume" section every session.
Last updated: **2026-06-12 (second session — UX/pipeline upgrade wave implemented)**.

## Goal

Photo(s) of one subject → high-quality 3D gaussian splat + mesh, fully on-device
(Swift/MLX, macOS). Current push: **multi-view** (6-view sets or one AI contact sheet →
one model) plus a UI/feature upgrade wave: per-image auto settings (done), per-view
progress tree, thumbnail expansion, working stop/cancel, responsive viewport progress,
an explicit 2K input policy, SAM3 landmark grounding with a semantic landmark list UI,
Sapiens2 diagnostics, and (researched, likely deferred) optional text prompt guidance.

## Current implementation status (verified 2026-06-12, second session)

- **Working & wired:** full single-image pipeline (low-light → Real-ESRGAN → RMBG/Vision
  → DINOv2 → DiT+CFG → voxel VAE → gaussian decoder → splat/mesh/point cloud); multi-view
  conditioning (multidiffusion/stochastic/concat) + yaw/IoU scoring; contact-sheet split;
  HD photo-texture backprojection; Metal 3DGS + SceneKit viewers; first-run installer.
- **New this session (built + CLI-verified, in-app visuals not yet observed live):**
  per-view auto settings rework (subject bbox/texture/orientation metrics, D-score
  steps formula with real spread — was pinned at 45); per-view progress tree +
  thumbnail lightbox; functional stop/cancel (immediate vs finish-candidate);
  responsive live cloud (per-step cadence late, progress shading, shimmer toggle,
  auto-rotate default off); 2K subject policy with alpha-preserving cascade
  (Decision 006); run metadata JSON per run.
- **Weights:** full engine weights are in `~/Library/Application Support/LiToStudio/
  weights/` (the app generates!). Repo `weights/` has only metallib + RealESRGAN —
  note `./run.sh` points `LITO_WEIGHTS_DIR` at the repo dir, bypassing App Support.
  `RMBG2.mlpackage` + `SapiensNormal.mlpackage` still absent; `testset/` still off-repo
  → bench still can't run.
- **New (session 3, 2026-06-12):** taxonomy attached → `docs/LANDMARK_TAXONOMY.txt`;
  landmark grounding scaffold (`LandmarkGrounding.swift`: segmenter protocol with
  honest unavailable backend, view labels user>filename>pose, per-run
  `<base>_landmarks.json` with priors-based visibility matrix — `consumedByGenerator:
  false`); Semantic Landmarks panel (matrix + label correction + backend diagnostics);
  Vision framing classifier + raised-hand pose features; viewer Export menu (splat /
  mesh / metadata / landmark package) + mesh-quality caption; optional prompt field
  (metadata-only, clearly labeled). First live 6-view runs confirmed by user
  (tree/stop/analyzer working; texture skipped — RMBG absent ⇒ no cutouts).
- **New (session 4, deep — REAL model backends):** `tools/backend/` Python workers
  (uv venv, torch 2.12 + transformers 5.12, MPS) bridged via
  `LiToKit/PythonBackend.swift`, run pre-engine (no MLX memory overlap).
  **RMBG-2.0: WORKING** (license accepted, weights cached; real cutouts persisted as
  run artifacts; CoreML conversion blocked by deform_conv2d — worker is the path).
  **Sapiens2-pose-0.4b: WORKING** (real Goliath-308 keypoints per view → package +
  metadata + UI; subject-box crops, not RTMDet). **SAM 3.1: WORKING NATIVELY** —
  community CoreML conversion (`weights/sam3-coreml/`, AllanVester/SAM3.1-CoreML-FP16,
  ungated) + `Sam3CoreML.swift` driver, preferred over the still-gated python worker;
  CLIP-BPE prompt tokens baked; verified by visual mask inspection on testset photos
  (Decision 012). Mesh: component cleanup added (islands < 2 %). Verify via
  `LiToSmoke ground <imgs>` / `LiToSmoke sam3`.
- **Missing entirely:** text conditioning (prompt field records to metadata only).
- **Blocked:** Sapiens *normal* model for mesh refinement still needs the user's
  Colab conversion (`SapiensNormal.mlpackage` — separate from the pose model).

## Known issues

- The app instance running before this session still has the old binary — relaunch to
  get the new UI (Stop button, progress tree, lightbox, new analyzer panel).
- Live-cloud decode now fires up to every step late in sampling; cost on a 16 GB
  machine under 6-view multidiffusion is unmeasured — if runs swap, back the cadence
  off in `LiToEngine` (`onStepSample` gate).
- weights-v1 GitHub release 404s — fresh installs can't fetch metallib/RealESRGAN until
  the user runs `gh release create`.
- Multi-view engine path still has no end-to-end bench (testset off-repo).

## Desired UX (target)

Drop up to 6 views (or one sheet) → auto-analyzed settings → per-view progress tree with
expandable thumbnails → live dot/splat assembly in the viewport → Stop button that
actually halts compute → results as splat + mesh, with SAM3-grounded landmark list
(neutral academic labels from the taxonomy file) and Sapiens2 diagnostics per run.

## Blocked / unknown

- `SapiensNormal.mlpackage` — user's Colab conversion (sapiens2 0.8b).
- SAM3 — no CoreML/MLX conversion exists in this repo; integration is design-only
  (`docs/LITO_TECHNICAL_NOTES.md` § SAM3). **Taxonomy txt not yet attached** — landmark
  label set is pending; do not invent labels.
- Text prompt guidance — not supported by the current checkpoint; see
  `docs/LITO_PROMPT_GUIDANCE_RESEARCH.md`.
- 2K input policy — needs a decision (current: Real-ESRGAN upscales to max dim 4096;
  conditioning is always 518²).

## Last worked on / how to resume

- **2026-06-12 (session 4, deep):** real model backends installed + wired (see status
  above). Verified: `tools/backend/setup.sh` (venv ok), workers standalone AND through
  the Swift adapters (`LiToSmoke ground` on testset photos: 2/2 RMBG cutouts, real
  Sapiens2 keypoints); `swift build` debug+release clean.
- **2026-06-12 (session 4c — SAM 3.1 finalized, Decision 013):** fixed the three
  issues from the first SAM3 integration — (1) whole-person masks → person-silhouette
  gating (`personMask288` from the RMBG cutout) rejects the fallback + strips speckle;
  (2) presence floor raised to 0.51 (real cross-view discrimination: back view has no
  face/navel/chain, front does); (3) the unreadable raw-mask display → region-over-
  photo **overlays** (`writeOverlay`, fixed an upside-down flip bug). Plus the text
  field now drives SAM 3.1 (`ClipTokenizer` worker → `groundConcept` → package
  `userConcept`; "leather glove"/"cargo pants" verified). Builds debug+release clean;
  app relaunched. NOT verified: a fresh in-app 6-view run exercising the overlays +
  text concept live; parity vs official facebook/sam3 (gated). Uncommitted.
- **2026-06-12 (session 4b):** SAM 3.1 gate bypassed — adopted the community CoreML
  conversion (user's find), native `Sam3CoreML` driver + pipeline preference.
- **2026-06-12 (session 3):** Sections 6–10 — landmark grounding scaffold + package
  export, Semantic Landmarks panel, Vision framing/raised-hand features, viewer
  Export menu + mesh caption, metadata-only prompt field.
- **2026-06-12 (session 2):** Sections 1–5 (auto-settings rework, progress tree +
  lightbox, stop/cancel, responsive viewport, 2K policy + run metadata) — since
  user-verified live (screenshots: tree, per-view analyses, 6-view runs end-to-end).
- **2026-06-12 (session 1):** created this context system.
- Next: relaunch + eyeball the Semantic Landmarks panel and Export menu on a 6-view
  run; fetch RMBG2 to unlock photo texture; bench multi-view vs single
  (`~/Downloads/testset/` exists); SAM3 backend research (checkpoint/license/
  conversion route) when ready.
- To resume: read this file → `LITO_TODO.md` → `LITO_UPGRADE_PLAN.md`; check both
  `weights/` and `~/Library/Application Support/LiToStudio/weights/` before assuming
  the engine can/can't run; check `LITO_DECISIONS.md` before changing defaults/labels.
