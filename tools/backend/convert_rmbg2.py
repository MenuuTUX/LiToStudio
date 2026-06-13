#!/usr/bin/env python3
"""RMBG-2.0 (BiRefNet) -> CoreML RMBG2.mlpackage, matching the interface
Sources/LiToKit/RMBG.swift expects:

  input : float32 multiarray (1, 3, 1024, 1024), ImageNet-normalized
          (Swift does the normalization itself, so NO normalization is baked in)
  output: float32 multiarray (1, 1, 1024, 1024) probability mask in [0, 1]
          (Swift auto-detects probability vs logits; we emit sigmoid)

Requires: HF account with the briaai/RMBG-2.0 license accepted + `hf auth login`.
Usage:  .venv/bin/python convert_rmbg2.py [output_dir]   (default: ../../weights)
"""
import sys, os, traceback

def main():
    out_dir = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                              os.path.join(os.path.dirname(__file__), "..", "..", "weights"))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "RMBG2.mlpackage")

    import torch
    from transformers import AutoModelForImageSegmentation

    print("▶ Loading briaai/RMBG-2.0 (gated — needs accepted license + hf auth)…", flush=True)
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-2.0", trust_remote_code=True)
    except Exception as e:
        print("✗ Could not load briaai/RMBG-2.0:", e)
        print("  → Accept the license at https://huggingface.co/briaai/RMBG-2.0 and run `hf auth login`.")
        sys.exit(2)
    model.eval()

    class MaskHead(torch.nn.Module):
        """BiRefNet returns a list of decoder stages; the last is the refined mask."""
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            preds = self.m(x)
            if isinstance(preds, (list, tuple)):
                preds = preds[-1]
            if hasattr(preds, "logits"):
                preds = preds.logits
            return torch.sigmoid(preds)

    wrapped = MaskHead(model).eval()
    example = torch.zeros(1, 3, 1024, 1024)

    import coremltools as ct
    # torch.export with a static input shape resolves BiRefNet's dynamic-shape
    # integer casts that break the legacy torch.jit.trace conversion path.
    print("▶ Exporting (torch.export, static 1024²)…", flush=True)
    with torch.no_grad():
        try:
            exported = torch.export.export(wrapped, (example,))
            source_model = exported.run_decompositions({})
        except Exception as e:
            print(f"  torch.export failed ({e}); falling back to jit.trace", flush=True)
            source_model = torch.jit.trace(wrapped, example, strict=False)

    print("▶ Converting to CoreML (this takes a few minutes)…", flush=True)
    mlmodel = ct.convert(
        source_model,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 1024, 1024), dtype=float)],
        outputs=[ct.TensorType(name="mask")],
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,   # RMBG.swift forces fp32 GPU compute
        convert_to="mlprogram",
    )
    mlmodel.short_description = "RMBG-2.0 (BiRefNet) background matting — input ImageNet-normalized (1,3,1024,1024), output probability mask (Bria license: commercial use requires a Bria agreement)"
    mlmodel.save(out_path)
    print(f"✓ wrote {out_path}", flush=True)

    # Smoke check: load + run a random input through CoreML, verify mask range.
    import numpy as np
    from coremltools.models import MLModel
    m = MLModel(out_path)
    rnd = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
    out = m.predict({"input": rnd})
    arr = next(iter(out.values()))
    print(f"✓ CoreML smoke: output shape {arr.shape}, range [{arr.min():.3f}, {arr.max():.3f}]")
    if not (0.0 <= arr.min() and arr.max() <= 1.0):
        print("✗ output not a probability mask — conversion is wrong"); sys.exit(3)

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)
