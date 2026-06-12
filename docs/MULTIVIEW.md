# Multi-view → one HD model — plan

> **Status 2026-06-11:** M1, M2, M4 and contact-sheet ingestion are IMPLEMENTED
> (`SheetSplit.swift`, `Dit.swift` multi-cond sampling, `LiToEngine.swift` yaw/IoU
> scoring, `TextureProject.swift` backprojection; wired through the app and
> `LiToSmoke engine|sculpt|split|texture`). Sheet splitting and texture projection
> are self-tested (synthetic sheet from testset photos; synthetic two-view sphere —
> hemispheres take the right photo, zero occlusion bleed). The DiT/engine multi-view
> path builds but needs the weights re-downloaded (weights/ is currently absent) for
> an end-to-end run + bench. M3 (multi-yaw NormalRefine) still pending Sapiens
> .mlpackage. Workflow: ask a frontier image model for a character turnaround sheet
> (same subject, same pose, 4–6 angles, plain background), drop the single sheet —
> the app OCR-erases labels, splits figures by bounding box, and conditions one
> model on all views.

Goal: accept N photos of the *same subject in the same pose* from different angles
(front / ¾ / side / back) and use all of them to produce one model that is sharper and
more correct than any single-image run — especially the back and sides, which today are
pure generative guesses.

## Why this is feasible without retraining

The DiT conditions on DINOv2 tokens via **unmasked cross-attention with arbitrary token
count m** (`Dit.swift::crossAttn`). Conditioning is pose-free (no camera embedding), so
multi-image conditioning is a *sampling-time* change, exactly like TRELLIS's
`run_multi_image` modes. The latent (8192×32 → 64³ occupancy) caps raw geometric detail,
so "really high definition" must come from two places:

1. **Better geometry** — multi-view conditioning + multi-view seed scoring (M1).
2. **Measured detail on top** — per-view Sapiens re-sculpting and HD photo texture
   backprojection (M3/M4). This is where the 4096px upscaled photos actually pay off;
   the 518² cond crop never sees that resolution.

## M1 — multi-view sampling in the engine (core, ~1–2 days)

`LiToEngine.generate` takes `imageURLs: [URL]`; per-view preprocessing reuses the whole
existing chain (low-light, Real-ESRGAN, RMBG, person-trim, cond crop — bbox-normalized
crop already makes full-body scale consistent across views). DINOv2 encodes each view →
`conds: [MLXArray]` (1374×2048 each). `DiT.sample` grows a mode switch:

- **multidiffusion** (default): per Heun half-step, average the conditional velocities
  over views, then CFG once — by linearity
  `mean_i[vu + cfg(vc_i − vu)] = vu + cfg(mean_i vc_i − vu)`, so the uncond eval stays
  **single** regardless of N. Cost per step: N+1 evals vs 2 today (N=4 → 2.5× sampling
  time). Keep the existing eval-between-forwards pattern — never batch views into the
  batch dim on 16 GB.
- **stochastic**: round-robin one view per half-step. Same cost as single-view, noisier;
  useful fallback for many views / low memory.
- **concat**: `cond = concat(views, axis: 1)` → (1, N·1374, 2048), single pass.
  Off-distribution (trained single-image) but worth a flag — and it becomes the *right*
  mode after the fine-tune (see below).

`PipelineArgs.imagePath` → `imagePaths: [String]`; single image stays the degenerate case.

## M2 — multi-view scoring + yaw estimation (~1 day)

- **Yaw per view**: the cond camera is orthographic along +x (u=y, v=z). For each view,
  sweep yaw 0–360° (5° grid) rotating occupancy coords around z and take the
  silhouette-IoU argmax against that view's alpha. Anchor: user-tagged front = 0°, or
  highest-IoU view wins front. UI lets the user override tags (front/back/left/right/free).
- **Multi-view seed selection**: `silhouetteIoU` → mean IoU over all views at their
  estimated yaws. Same seed-candidate loop, much stronger signal than one view — this is
  the cheapest big win in the whole plan.
- **Consistency check**: per-view IoU spread doubles as a pose-consistency warning. The
  model emits ONE rigid shape; if the subject's arms moved between shots, multidiffusion
  averages the conflict into mush. Warn when a view's best IoU lags the mean by >0.1 and
  offer to drop or down-weight it.

## M3 — multi-view Sapiens re-sculpting (~2–3 days)

`NormalRefine` is pinned to the ±x camera. Parameterize the raster/projection by yaw
(rotate mesh into view frame before `Raster.draw`), then per view: Sapiens normals →
screened-Poisson depth solve → sculpt the *view-facing* surface, plus per-view silhouette
snap. Blend overlapping deltas by cosine-to-camera weight. The back photo finally sculpts
the back of the mesh. (Sapiens .mlpackage still pending the Colab conversion — the
silhouette-snap half of this works without it.)

## M4 — HD texture backprojection (~1–2 days)

Project every gaussian center / mesh vertex into each view (orthographic, per-view yaw),
z-buffer visibility from the rasterized mesh, sample the **4096px upscaled** photo, blend
across views with `w = max(0, n·view)^k · alpha`. Replace generated splat colors and bake
mesh vertex colors (later: UV texture + displacement, per the sculpt roadmap). This is
the single biggest "looks real" step — generated colors are soft 64³-latent guesses;
backprojection puts actual pixels on the surface.

## M5 — optional ceiling

- **Fine-tune addendum** (`FINETUNE_HUMANS.md`): when rendering condition views per scan,
  train with random 1–4 view **concat** conditioning. Makes M1's concat mode
  in-distribution → best multi-view quality, single-pass cost.
- Differentiable splat refinement against the N photos (MLX renderer) — heavy, last.

## Validation

Bench protocol per `bench/EXPERIMENTS.md`. Add a multi-view testset (≥3 subjects ×
4–6 views). Killer metric: **held-out-view IoU** — condition on N−1 views, score the
silhouette against the held-out view. Compare single-image baseline vs stochastic vs
multidiffusion vs concat at matched wall-clock (25 steps, cfg 3, seed 7, best-of-3).

## Order

1. M1 + M2 (engine + scoring) — geometry win, ships alone.
2. M4 texture backprojection — biggest visible realism win, independent of Sapiens.
3. M3 multi-view sculpt — once SapiensNormal.mlpackage lands.
4. M5 fine-tune — folds into the already-planned Colab human fine-tune.
