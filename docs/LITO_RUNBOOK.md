# LiTo runbook

Commands verified against `run.sh`, `Package.swift`, `SETUP.md`, and CI on 2026-06-12.

## Install / setup
- `./run.sh` — builds (release) and launches the app; on a machine without weights the
  app shows first-run setup (downloads ~7.8 GB, converts the LiTo ckpt locally).
  **Note:** the weights-v1 GitHub release currently 404s, so the metallib/RealESRGAN
  downloads fail on a truly fresh machine until the release is recreated; this checkout
  already has both files in `weights/`.
- Manual/offline install: see `SETUP.md` (incl.
  `swift run -c release LiToConvert lito_dit_rgba.ckpt weights/lito.safetensors`).
- Weights dir resolution: `weights/` next to the checkout, or `LITO_WEIGHTS_DIR`, or
  `~/Library/Application Support/LiToStudio/weights` (see `Config` in PipelineRunner.swift).

## Dev / run
- `./run.sh` — release build + launch. `./run.sh debug` — debug build + launch.
- Xcode: `xcodegen generate` (project.yml) → open `LiToStudio.xcodeproj`. The post-build
  step colocates `mlx.metallib`.
- Plain `swift build` works but the resulting binary won't launch the GPU backend or
  splat viewer (metallib colocation + MetalSplatter shader compile) — use `run.sh`.

## Test
- **There is no test target.** `swift test` runs zero tests (CI's test step is a no-op).
- Actual verification is the smoke/dev harness (requires full weights):
  - `./run.sh smoke` — MLX + weights sanity check, no UI.
  - `LiToSmoke` subcommands (built to `.build/<config>/LiToSmoke`): per-stage parity
    (`dino`, `dit`, `voxel`, `gs`, `rmbg`, `cond`), `split`, `texture`, `mesh`,
    `refine`, `normals`, `render`, `score`, `gscheck`, `coreml`. See `SETUP.md`.
  - No-weights checks: `LiToSmoke analyze <img[,img2,…]>` (auto-settings working
    shown, incl. orientation/framing/raised-hand), `LiToSmoke upscale
    weights/RealESRGAN_x4.mlmodel <img> [out.png]` (2K-policy cascade + alpha
    preservation), `LiToSmoke landmarks <img[,img2,…]> [out.json]` (priors-only
    package), and `LiToSmoke ground <img[,img2,…]> [outDir]` (REAL backends through
    the app's adapters: RMBG cutouts, Sapiens2 pose, SAM3 when its weights exist).
    The local testset lives at `~/Downloads/testset/`.

## Model backend (Python workers)
- `tools/backend/setup.sh` — create the uv venv; `setup.sh check` — status of venv +
  each model's HF cache. Full install/auth steps per model: `tools/backend/README.md`.
- Current state (2026-06-12): RMBG-2.0 working (PyTorch worker — CoreML conversion
  blocked by deform_conv2d); Sapiens2-pose-0.4b working; **SAM 3.1 working natively**
  via the CoreML packages in `weights/sam3-coreml/` (no gate; re-download with
  `hf download AllanVester/SAM3.1-CoreML-FP16 --local-dir weights/sam3-coreml`, then
  regenerate prompts with `tools/backend/.venv/bin/python tools/backend/make_sam3_tokens.py`).
  The official facebook/sam3 (gated) is only wanted as a parity check now.
- SAM3 CoreML checks: `LiToSmoke sam3 weights/sam3-coreml <img> [norms|all|L001..]`
  (norm sweep / all-prompt grounding with mask PNGs to /tmp).

## Lint / typecheck
- No lint config exists (no SwiftLint/SwiftFormat files). `swift build` is the
  typecheck; it must pass warning-free-ish (Swift 6 strict concurrency is on).

## Build
- `swift build -c release` (engine + app), or `./run.sh` which also handles the two
  metallib steps. CI mirrors `swift build -v` on macos-latest.

## Manual feature verification
- **Six-view upload:** launch app → drag 6 images (same subject/pose, different angles)
  onto the drop zone → thumbnail grid shows "6 views — multi-view conditioning" →
  Generate. CLI equivalent: `./run.sh engine "v1.png,…"` — actually `LiToSmoke engine
  weights "v1.png,v2.png,…"` takes a comma-joined list and prints per-view yaw/IoU.
  Contact sheet: drop one sheet image; `[Sheet]` log lines show the split.
- **Progress tree:** implemented 2026-06-12 — verify: drop N views, Generate → sidebar
  shows one chip row per view (original → 4× → bg → crop → dino → sapiens → sam3 →
  token) with Sapiens/SAM3 visibly unavailable, plus the trunk checklist (conditioning
  → candidates with live step progress → decode → texture → mesh → result). Click any
  thumbnail → lightbox (zoom/pan, Esc closes). Tree persists after the run.
- **Stop/cancel:** implemented 2026-06-12 — verify: press Stop mid-sampling early →
  GPU work ends within ~one step, orange "Stopped" card, last dot cloud stays in the
  viewport, temp files removed, no partial files in results/. Press Stop at ≥80 % of a
  candidate → dialog offers "Finish candidate, then stop" (produces a real splat,
  texture/mesh skipped) vs "Stop immediately".
- **Viewport progress:** start a generation and watch white occupancy dots appear and
  densify every few sampling steps (needs `onStepCloud` path; dots come from real
  intermediate voxel decodes, so the first emissions may legitimately be empty).
- **Run metadata export:** implemented 2026-06-12 — every completed run writes
  `<base>_run.json` next to the artifacts in
  `~/Library/Application Support/LiToStudio/results/` (settings, seedUsed, per-view
  yaw/IoU, artifact names, full auto-detect analysis when auto was on).

## Bench (quality)
- `bench/run_baseline.sh <outdir>` — 25 steps / cfg 3 / seed 7 / best-of-3 per photo,
  ~1.5–2.5 h per photo. Requires full weights and the off-repo `testset/`. Log results
  in `bench/EXPERIMENTS.md` (Decision 007).
