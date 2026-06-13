# LiTo decision log

Add new decisions here instead of rediscovering them. Format: number, title, decision,
reason, status (active / superseded by NNN).

## Decision 001 — No fabricated model output
**Decision:** Never fake or stub the *output* of SAM3, Sapiens2, mesh conversion,
text conditioning, or any model. Absent models get a clean interface/adapter plus an
honest "missing/blocked" status in TODO + architecture map.
**Reason:** Fake outputs poison quality evaluation and waste bench time; the project's
whole verification protocol depends on measured results.
**Status:** active.

## Decision 002 — Neutral academic terminology for landmarks/labels
**Decision:** All segmentation/landmark labels, UI strings, code identifiers, and docs
use neutral academic anatomical and garment terminology, sourced from the user's
landmark/object taxonomy txt file. No casual or sexualized wording.
**Reason:** Professional, unambiguous vocabulary; consistent with the scientific
SAM3-grounding design; explicit user requirement.
**Status:** active — taxonomy attached 2026-06-12 and canonical at
`docs/LANDMARK_TAXONOMY.txt`; the core set (L001–L012) + per-view priors are embedded
in `Sources/LiToKit/LandmarkGrounding.swift`. All labels in code/UI come from it.

## Decision 003 — Progress must show all view branches
**Decision:** Multi-view runs display one progress branch per input view (up to 6) —
preprocess/encode per view, then the shared sampling/decode trunk — instead of a single
interleaved stage string.
**Reason:** With 6 views the single bar is unreadable and hides which view failed/skipped
a stage (e.g. RMBG fallback on one view only).
**Status:** active — not yet implemented (TODO Phase B).

## Decision 004 — Stop/cancel must actually halt compute
**Decision:** Cancellation is cooperative and threaded into the engine's step loop
(checked at least once per DiT half-step and between pipeline phases). Cancelled runs
delete temp files and never write partial results. UI gets a Stop button.
**Reason:** Today `cancel()` only detaches the UI; a 6-view best-of-N run keeps the GPU
busy for tens of minutes with no way to reclaim it.
**Status:** implemented 2026-06-12 (`GenCancelToken`; immediate vs finish-candidate;
see TECHNICAL_NOTES § Cancellation). Live verification in the app still pending.

## Decision 005 — Auto-rotation is a user toggle, default OFF
**Decision:** Viewer auto-rotation stays toggleable in the viewport; the default is
**off** (amended 2026-06-12 on user request — manual orbit is the primary interaction,
including during generation via the live-cloud view).
**Reason:** Useful for showcase, annoying for inspection; both needs are real.
**Status:** active (toggles in `ViewerPane` for results and the running overlay for the
live cloud; `AppModel.autoRotate = false` default).

## Decision 006 — 2K input policy: subject long side ≥ 2048 px
**Decision (2026-06-12):** "2K input quality" means the *estimated subject* (foreground
bbox) long side reaches ≥ 2048 px, not the canvas. Skip when already there and sharp;
warn (don't upscale) when the canvas is at the 4096 px cap with a small subject; up to
two Real-ESRGAN passes otherwise, alpha-preserving, with a source-limited warning when
even the cascade can't reach the target. Canvas hard cap stays 4096 px (16 GB memory
budget).
**Reason:** Conditioning crops to the subject (518²) and texture backprojection samples
subject pixels — background canvas never reaches the model, so a canvas-based threshold
optimizes the wrong quantity.
**Status:** active — implemented in PipelineRunner + ImageAnalyzer + Upscaler; verified
via `LiToSmoke upscale` (see TECHNICAL_NOTES § 2K image policy).

## Decision 007 — Quality changes go through the bench protocol
**Decision:** Any change claiming better output quality runs `bench/run_baseline.sh`
and logs a row in `bench/EXPERIMENTS.md` before being believed.
**Reason:** Seed luck exceeds most real improvements; single-photo checks mislead.
**Status:** active (pre-existing project rule, recorded here).

## Decision 008 — Text prompt guidance is deferred (for DiT geometry)
**Decision:** Do not attempt text *conditioning of the DiT* against the current LiTo
checkpoint; it has no text pathway. The optional text field instead drives SAM 3.1
concept segmentation (real text → region in the landmark package — Decision 013), which
is honest and never touches generated geometry.
**Reason:** No text encoder exists in the checkpoint; bolting one on risks destroying
visual identity (the project's core value). SAM3 gives the text field a real, safe job.
**Status:** active (DiT text-conditioning still deferred; text→SAM3 grounding shipped).

## Decision 009 — Python tooling allowed under tools/ (for now)
**Decision:** The pure-CoreML/no-Python restriction is suspended (2026-06-10): quality
first, Python conversion tooling is fine under `tools/`; nativize later once results
are good. Runtime app remains Python-free.
**Reason:** Conversion velocity matters more than purity while models are in flux.
**Status:** active.

## Decision 010 — Landmark package: priors are never presented as detections
**Decision (2026-06-12):** The per-run landmark package and the Semantic Landmarks UI
distinguish three things that must never blur: taxonomy *expectations* per view label
(§ L priors), Apple Vision *pose features* (orientation/framing/raised-hand), and
*detections* (only from a real grounding backend — none installed). View-label
precedence: user override > filename token > pose estimate > unknown. The package
ships `consumedByGenerator: false` until a conditioning fine-tune exists; the DiT's
`sample(conds:)` token-stream list is the agreed integration seam.
**Reason:** Decision 001 (no fabricated model output) applied to grounding: priors
look like detections if the UI is careless, and that would poison every downstream
quality judgment.
**Status:** active.

## Decision 011 — Heavy model backends run as Python worker processes
**Decision (2026-06-12):** RMBG-2.0, SAM3, and Sapiens2 run via `tools/backend/`
(one uv venv, torch/transformers on MPS); the Swift app spawns one worker process per
batch through `PythonBackend.swift` and consumes JSON + image artifacts. Workers exit
before MLX sampling starts (no memory overlap on 16 GB). Sub-decisions:
- RMBG-2.0 **CoreML conversion is blocked** — BiRefNet uses
  `torchvision::deform_conv2d`, unsupported by coremltools ≤ 9.x. The PyTorch worker
  is the active path; `convert_rmbg2.py` stays for a future coremltools.
- Sapiens2 person crops come from the app's real subject-box estimate instead of the
  upstream RTMDet detector (avoids the mmdet/mmcv stack); slightly looser crops,
  honest and documented.
- Sapiens2 keypoint names: Goliath-308 = the 344 Goliath keypoints minus 36 teeth
  (sapiens dataset config), shipped as `tools/backend/goliath_keypoints.json`.
**Reason:** "Real model output first" (this stage's mandate) beats native purity;
Decision 009 already allows Python under tools/. Nativize later if conversion paths
open up.
**Status:** active (SAM3 nativized same day — see Decision 012).

## Decision 012 — SAM 3.1 via the community CoreML conversion (native, preferred)
**Decision (2026-06-12):** adopt `AllanVester/SAM3.1-CoreML-FP16` (ImageEncoder +
TextEncoder + Detector, fp16, ungated) as the PRIMARY SAM3 backend
(`weights/sam3-coreml/`, driver `Sources/LiToKit/Sam3CoreML.swift`); the gated
facebook/sam3 Python worker stays as the fallback path.
**Findings that make it workable:** (1) full text-prompted stack incl. text encoder;
(2) tokenizer = CLIP BPE — confirmed empirically (baked token IDs in
`prompt_tokens.json`; wrong tokens could not produce semantically correct masks);
(3) input normalization is baked into the conversion (detections identical from raw
0–255 to CLIP-normalized inputs — we standardize on CLIP); (4) bbox derived from the
thresholded mask, sidestepping the undocumented box-head format; (5) Detector fails
ANE compilation → all three models load CPU+GPU; (6) scores cluster low
(~0.50–0.65) → backend threshold 0.5 and confidences always surfaced.
**License:** the SAM License is a custom permissive license (commercial + research);
the mirror is plausibly legitimate redistribution, but provenance/fidelity rests on
the uploader — verified empirically on test photos, not against the official
implementation (gated). Re-verify against facebook/sam3 once access is approved.
**Status:** active — verified on testset photos (visual mask inspection: face/hair/
garment masks correct; 12 prompts ≈ 6.4 s per image after one-time model compile).
Finalized by Decision 013.

## Decision 013 — SAM 3.1 finalization: person-mask gating, overlay display, text concept
**Decision (2026-06-12):** (1) gate detections with the RMBG person silhouette —
intersect every mask with it and reject part-concepts that fill > 70 % of the person
(the whole-person fallback); honest `not_detected` over a wrong region. (2) Presence
floor 0.51 (`LITO_SAM3_THRESHOLD`), measured from the kept/rejected score gap on real
photos. (3) Show detections as a highlight **overlay over the original photo**, not the
raw binary mask. (4) The optional text field, with SAM 3.1 installed, becomes a live
concept segmented per view (CLIP tokenizer worker → `groundConcept` → package
`userConcept`); it never alters DiT geometry (Decision 008).
**Reason:** the first integration showed whole-person masks at floor confidence and an
unreadable raw-mask display (both user-reported); the text field did nothing.
**Status:** active — verified via `LiToSmoke sam3 / sam3concept / ground` (back vs front
cross-view discrimination correct; "leather glove" / "cargo pants" land on the right
regions; overlays right-side-up).
