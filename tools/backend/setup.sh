#!/usr/bin/env bash
# LiTo Studio model backend — one Python venv for RMBG2 conversion, the SAM3
# landmark worker, and the Sapiens2 pose worker. Requires `uv` (https://docs.astral.sh/uv).
#
#   ./setup.sh            create .venv and install everything
#   ./setup.sh check      print backend + model availability status
set -euo pipefail
cd "$(dirname "$0")"

if [[ "${1:-}" == "check" ]]; then
  [[ -x .venv/bin/python ]] && echo "venv: OK ($(.venv/bin/python -V))" || { echo "venv: MISSING — run ./setup.sh"; exit 1; }
  .venv/bin/python - <<'EOF'
import importlib, os
for m in ["torch", "transformers", "coremltools", "PIL", "huggingface_hub"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m}: {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"{m}: MISSING ({e})")
hub = os.path.expanduser("~/.cache/huggingface/hub")
for repo in ["models--briaai--RMBG-2.0", "models--facebook--sam3", "models--facebook--sapiens2-pose-0.4b"]:
    print(f"{repo}: {'cached' if os.path.isdir(os.path.join(hub, repo, 'snapshots')) else 'not downloaded'}")
EOF
  exit 0
fi

echo "▶ Creating venv (python 3.12)…"
uv venv --python 3.12 .venv
echo "▶ Installing requirements…"
uv pip install --python .venv/bin/python -r requirements.txt
echo "▶ Sanity check…"
.venv/bin/python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| mps:', torch.backends.mps.is_available())"
echo "✓ backend venv ready. Next: ./setup.sh check, then see README.md for model installs."
