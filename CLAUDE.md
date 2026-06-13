# LiTo Studio — repo instructions

Native macOS app (Swift 6 / SwiftUI / MLX-Swift) that runs Apple's LiTo image-to-3D
model fully on-device: photo(s) → gaussian splat + triangle mesh. No Python at runtime.

## Read first, in this order

1. `docs/LITO_CONTEXT_PACK.md` — project memory: goal, status, how to resume.
2. `docs/LITO_TODO.md` — what's done / in progress / blocked.
3. `docs/LITO_ARCHITECTURE_MAP.md` — where every system lives, with honest status.
4. `docs/LITO_DECISIONS.md` — settled decisions; do not re-litigate them.
5. `docs/LITO_RUNBOOK.md` — build/run/verify commands.

## Always-follow rules

- **Never fake model output.** No placeholder tensors pretending to be SAM3, Sapiens2,
  text-conditioning, or mesh results. If a model/weight is absent, build a clean
  interface and document it as *missing* or *blocked* in the TODO + architecture map.
- **Don't hallucinate APIs.** Check the actual code (`Sources/`) before claiming a
  function, mode, or weight exists. Weights are gitignored — verify `weights/` contents
  before assuming the engine can run.
- **Quality claims need the bench.** Single-photo spot checks lie (seed luck). Use
  `bench/run_baseline.sh` + `bench/EXPERIMENTS.md` protocol for anything that claims
  to change output quality.
- **Conventions:** z-up world, xyzw quaternions, reference sampling = 20 steps / CFG 3,
  cond camera at +x looking −x. `mlx.metallib` must sit next to the executable
  (`./run.sh` and the Xcode post-build handle it).
- **Terminology:** segmentation/landmark labels and all user-facing text use neutral
  academic anatomical/garment terminology (see the landmark taxonomy file referenced
  in `docs/LITO_DECISIONS.md`). No casual or sexualized wording anywhere.
- Record new decisions in `docs/LITO_DECISIONS.md`; update `docs/LITO_CONTEXT_PACK.md`
  ("last worked on" section) at the end of every working session.

## Verification

- `swift build` must pass (that is the type check; there is no separate lint config).
- There is **no test target** — do not claim "tests pass". Self-test via
  `./run.sh smoke` (needs weights) and the `LiToSmoke` subcommands.
- End-to-end runs currently require re-downloading weights (see context pack).

## Final response format

End substantial work with: what changed (files), what was verified and how, what is
untested or blocked and why, and which docs were updated.
