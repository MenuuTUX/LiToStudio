# Fine-tuning LiTo on humans (Colab) — runbook

Goal: close the human-geometry gap vs Tripo. LiTo trained on 240k generic Objaverse
objects; your inputs are people. Same model + human data = the real quality ceiling lift.
The paper's training configs ship in `apple/ml-lito` (`configs/lito/generator/lito_dit_8k32.yaml`).

## What gets fine-tuned
Only the **DiT** (623M params — fits a Colab A100 40GB in bf16 with small batch +
grad accumulation). The tokenizer stays frozen — it already reconstructs humans fine
(it's trained for *any* surface light field); the generative *prior* is what lacks humans.

## Pipeline (all offline, one-time per dataset)
1. **Data**: THuman2.1 (~2,500 clothed human scans, free for research) and/or 2K2K (2,050
   scans). Both are textured meshes/point clouds.
2. **Render** RGBD multiview per scan with Blender (150 views on a sphere — reuse
   `ml-lito`'s `dataset_toolkits`-style script; the paper renders with 3 lighting setups,
   1 is enough for fine-tuning).
3. **Tokenize**: run the frozen LiTo encoder (`pretrained_tokenizer.fpoint_encoder`,
   already in your checkpoint) over the rendered RGBD point clouds → 8192×32 latents.
   Cache them — this is the expensive pass, do it once.
4. **Condition images**: for each scan, render extra single views (random poses/lighting,
   ideally phone-camera-like FOV) → (image, latent) training pairs.
5. **Fine-tune** the DiT from `lito_dit_rgba.ckpt` with the stock config, lowered LR
   (~1e-5 → 2e-5), 20–50k iterations, EMA on. Monitor conditioning-view FID on a held-out
   split (the paper's Toys4k protocol, but on held-out humans).
6. **Convert** the EMA weights with `LiToConvert` (torch-zip → safetensors), drop the file
   into `weights/`, run the app. No app changes needed — same architecture, new weights.

## Colab practicalities
- A100 40GB: bf16, batch 4–8 with grad-accum ×4. The paper trained 600k iters from
  scratch; a domain fine-tune needs ~5% of that.
- Checkpoint to Drive every 30–60 min (Colab preemption).
- Total budget estimate: rendering ~1 GPU-day equivalent (CPU-heavy, can run locally
  overnight on the Mac in Blender), tokenizing ~hours, fine-tune ~1–2 A100-days.

## Risk notes
- Don't over-train: human-only fine-tuning will degrade generic objects. Keep a small
  Objaverse replay mix (~10–20%) if generic objects still matter.
- License: THuman/2K2K are research-only — fine for personal use, not for shipping weights.
