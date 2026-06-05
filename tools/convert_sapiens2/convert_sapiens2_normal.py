#!/usr/bin/env python3
"""Meta Sapiens2 surface-normal estimator → CoreML .mlpackage for LiToStudio.

Dev-side tool only (run via convert.sh, which builds the quarantined venv).
The app never runs Python — this produces the same artifact the Colab/Lightning
notebook would, plus a real on-device CoreML-vs-torch validation that the cloud
can't do (CoreML prediction only exists on macOS).

What gets baked into the converted graph (so Swift feeds raw pixels):
  input  : "image", RGB 768×1024 (W×H), values 0–255 (CoreML ImageType)
  prep   : ImageNet×255 mean/std normalization
  output : "normals", (1, 3, 1024, 768) float, unit-normalized
Channel convention is irrelevant downstream — NormalRefine self-calibrates axes.
"""
import argparse
import gc
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", default="0.8b", choices=["0.4b", "0.8b", "1b"])
    ap.add_argument("--out", required=True, help="output .mlpackage path")
    ap.add_argument("--skip-validate", action="store_true",
                    help="skip the CoreML-vs-torch check (saves ~2 GB RAM at the end)")
    args = ap.parse_args()

    import torch
    assert torch.__version__.startswith("2.5."), (
        f"torch {torch.__version__} loaded — expected 2.5.x; rerun convert.sh (it pins last)"
    )
    import coremltools as ct
    import torch.nn as nn
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download, list_repo_files

    # ── checkpoint ──────────────────────────────────────────────────────────
    repo = f"facebook/sapiens2-normal-{args.size}"
    files = [f for f in list_repo_files(repo) if f.endswith(".safetensors")]
    assert files, f"no .safetensors found in {repo}"
    print(f"▶ checkpoint {repo}/{files[0]} (cached under ~/.cache/huggingface)")
    ckpt = hf_hub_download(repo, files[0])

    # ── model from the repo config ──────────────────────────────────────────
    sys.path.insert(0, str(HERE / "sapiens2"))
    from sapiens.dense.src.models.init_model import init_model  # noqa: E402

    config = (HERE / "sapiens2/sapiens/dense/configs/normal/metasim_render_people"
              / f"sapiens2_{args.size}_normal_metasim_render_people-1024x768.py")
    print(f"▶ building model from {config.name}")
    model = init_model(str(config), ckpt, device="cpu").eval()

    # ── wrapper: normalization + upsample + unit-normalize in-graph ─────────
    class Wrapped(nn.Module):
        def __init__(self, m: nn.Module) -> None:
            super().__init__()
            self.m = m
            self.register_buffer("mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))

        def forward(self, x):                      # x: (1,3,1024,768) RGB in 0..255
            x = (x - self.mean) / self.std
            n = self.m(x)
            n = F.interpolate(n, size=(1024, 768), mode="bilinear", align_corners=False)
            return n / n.norm(dim=1, keepdim=True).clamp(min=1e-5)

    wrapped = Wrapped(model).eval()
    # Quantized example (exact uint8 values) so the validation comparison is apples-to-apples.
    example = (torch.rand(1, 3, 1024, 768) * 255).round()
    print("▶ tracing…")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example)
        ref = traced(example)
    assert tuple(ref.shape) == (1, 3, 1024, 768), f"unexpected output shape {tuple(ref.shape)}"

    del model, wrapped
    gc.collect()

    # ── CoreML conversion ────────────────────────────────────────────────────
    print("▶ converting to CoreML fp16 mlprogram (this is the slow part)…")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=(1, 3, 1024, 768),
                             scale=1.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="normals")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
    )
    mlmodel.user_defined_metadata["lito.normalized-input"] = "1"
    mlmodel.user_defined_metadata["lito.model"] = f"sapiens2-normal-{args.size}"
    mlmodel.short_description = (
        "Meta Sapiens2 surface-normal estimator wrapped for LiToStudio: "
        "RGB 0-255 in, unit normal map (1,3,1024,768) out. Sapiens2 License."
    )

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out))
    print(f"✓ saved {out}")

    # ── validation (macOS-only superpower: actually run the converted model) ─
    if args.skip_validate:
        return
    del traced
    gc.collect()
    print("▶ validating CoreML vs torch on the same input…")
    import numpy as np
    from PIL import Image

    pil = Image.fromarray(example[0].permute(1, 2, 0).numpy().astype("uint8"))
    pred = np.asarray(mlmodel.predict({"image": pil})["normals"], dtype=np.float32)
    a = ref.numpy()[0].reshape(3, -1)
    b = pred[0].reshape(3, -1)
    cos = (a * b).sum(0) / (np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0) + 1e-8)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    print(f"   mean angular diff {ang.mean():.3f}°, p95 {np.percentile(ang, 95):.3f}° "
          f"(fp16 conversion noise — anything ≲2° is healthy)")
    if ang.mean() > 5:
        print("✗ angular diff is suspiciously large — do NOT use this model; ping Claude with this output")
        sys.exit(1)
    print("✓ conversion validated")


if __name__ == "__main__":
    main()
