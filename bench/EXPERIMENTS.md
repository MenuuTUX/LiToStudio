# Quality experiment log

One entry per change that claims to improve sculpt quality. Protocol: run
`bench/run_baseline.sh bench/<experiment-name>` (same testset, 25 steps / cfg 3 /
seed 7 / best-of-3) and compare against `bench/baseline/` before believing anything.
Single-photo spot checks lie — seed luck is bigger than most real improvements.

What to compare per photo:
- silhouette IoU lines from the engine log (seed search prints them)
- mesh stats (verts/tris) and the render PNGs side by side
- `[refine]` line when Sapiens is active: Σ|corr|, moved verts, mean/max displacement
- wall-clock per stage (regressions count as costs)

| Date | Change | Branch/commit | Result vs baseline | Verdict |
|---|---|---|---|---|
| 2026-06-10 | Baseline (pre-Sapiens) | main @ 6295a5e+wip | — (reference) | running |
| | Sapiens2 normal refinement (pending .mlpackage) | | | |
| 2026-06-11 | Multi-view conditioning + photo texture (docs/MULTIVIEW.md) | wip | pending weights re-download; single-image path unchanged by construction (wrapper) | untested vs baseline |

Multi-view protocol addition: condition on N−1 views, score silhouette IoU against the
held-out view (`LiToSmoke engine weights "v1.png,v2.png,…"` prints per-view yaw/IoU).
Compare multidiffusion vs stochastic vs concat at matched wall-clock.
