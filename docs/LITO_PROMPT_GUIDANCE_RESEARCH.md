# Optional user text prompt guidance — research/status

Question: can a user text prompt steer LiTo generation (e.g. "longer coat", "smoother
hair") alongside the photo(s)? Status as of 2026-06-12: **not supported; deferred**
(Decision 008).

**UI status (updated 2026-06-12, Decision 013):** the optional "text concept guidance"
field now has a real, honest job when SAM 3.1 is installed: the phrase is segmented as
a SAM 3.1 concept in every view (CLIP tokenizer worker → `Sam3CoreML.groundConcept`)
and added to the landmark package as `userConcept` (region overlays shown in the
Semantic Landmarks panel). This is real **text → region** grounding. It still does NOT
condition the DiT — the checkpoint has no text pathway — so generated *geometry* is
never altered by the text, and the field copy + panel say so. Without SAM 3.1 the field
remains metadata-only (`userPrompt`). Empty field → no effect anywhere.

## Does the current pipeline support text conditioning?

**No.** Verified in code:
- The DiT (`Sources/LiToKit/Dit.swift`) cross-attends a single conditioning stream:
  DINOv2 *image* tokens (1374×2048 per view, from `Dinov2.swift`). There is no text
  encoder anywhere in `Sources/` (no CLIP/T5/tokenizer code; grep-verified).
- The unconditional branch for CFG uses a learned `y_embedding` (null-cond), not an
  empty text embedding.
- The checkpoint (`lito.safetensors`, converted from Apple's `lito_dit_rgba.ckpt`)
  ships no text-encoder weights; this is the *image-conditioned* model lineage
  (TRELLIS-image-style), not a text-conditioned variant.

## What would an integration need?

Any real text pathway requires new or different weights — there is no adapter that
makes the current checkpoint understand text:

1. **Text-conditioned checkpoint swap** — upstream TRELLIS has a text-large variant,
   but it is a different model (different conditioning dims, no photo input), so it
   replaces rather than augments identity-preserving image conditioning. Would need its
   own conversion (`LiToConvertCore`) + a text encoder port to MLX/CoreML.
2. **Dual-conditioning fine-tune** — fine-tune the current DiT to accept concatenated
   [DINOv2 tokens ‖ projected text tokens]. Integration point exists (cross-attention
   takes arbitrary token count m — the same property the multi-view concat mode uses),
   but this is a training project (Colab, datasets, new weights), not an app change.
3. **Prompt-as-preprocessing** — use text only *outside* the DiT: e.g. text-prompted
   segmentation (ties into the SAM3 plan) to mask/weight regions of the conditioning
   image, or generate an edited reference view with an external image model and feed it
   in as an extra view. No new LiTo weights needed; weakest steering, but honest.

If implemented, the seam in this codebase is `DiT.sample(conds:)` — conditioning is
already a list of token arrays, so an additional (projected) token stream slots into
`multidiffusion`/`concat` mechanics without restructuring.

## Risks

- **Identity override:** text tokens that conflict with the photo (option 1/2) pull the
  sample away from the subject's measured appearance — the project's core value is
  visual identity from photos; text is at best a tiebreaker for unseen regions.
- Off-distribution conditioning (bolting tokens onto an image-trained DiT without
  fine-tuning) degenerates output — same reason concat mode is flagged off-distribution.
- UX overpromise: a prompt box implies control the model doesn't have.

## Recommendation

Defer (Decision 008). If/when demanded: start with option 3 (text-prompted region
masking via the SAM3 grounding work — shared infrastructure, no new generative weights),
and fold option 2 into the already-planned human fine-tune (`FINETUNE_HUMANS.md`) only
if a concrete use case survives the multi-view + texture-backprojection quality wins.
