#!/usr/bin/env bash
# One-shot LOCAL conversion: Meta Sapiens2 normal estimator → weights/SapiensNormal.mlpackage
#
#   ./tools/convert_sapiens2/convert.sh            # default 0.8b (recommended for 16 GB)
#   ./tools/convert_sapiens2/convert.sh 0.4b       # lighter / faster
#   ./tools/convert_sapiens2/convert.sh 0.8b --force   # run even while the benchmark is going
#
# This is a QUARANTINED dev tool: it builds its own .venv here and never touches the
# app — LiToStudio itself stays pure Swift/CoreML with no Python at runtime.
# Everything it downloads/builds lands in this directory (gitignored) and the HF cache.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

SIZE="0.8b"; FORCE=0
for a in "$@"; do
  case "$a" in
    0.4b|0.8b|1b) SIZE="$a" ;;
    --force) FORCE=1 ;;
    *) echo "usage: convert.sh [0.4b|0.8b|1b] [--force]"; exit 1 ;;
  esac
done

# Conversion peaks ~8–10 GB RAM (fp32 weights + trace + convert). On a 16 GB machine
# that starves a running engine/benchmark — refuse unless forced.
if [[ $FORCE -eq 0 ]] && pgrep -f "LiToSmoke engine" >/dev/null; then
  echo "⚠️  A LiToSmoke engine run is active (the baseline benchmark?)."
  echo "   Conversion needs ~10 GB RAM and would starve it on a 16 GB machine."
  echo "   Wait for the current photo to finish, or rerun with --force."
  exit 1
fi

# Python 3.9–3.12 — the validated window for torch 2.5 + coremltools 8.3.
PY=""
for c in python3.12 python3.11 python3.10 python3.9 python3; do
  command -v "$c" >/dev/null 2>&1 || continue
  v="$("$c" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" || continue
  case "$v" in 3.9|3.10|3.11|3.12) PY="$c"; break ;; esac
done
if [[ -z "$PY" ]]; then
  echo "✗ No Python 3.9–3.12 found. Easiest fix:  brew install python@3.12"
  exit 1
fi
echo "▶ python: $PY ($("$PY" --version 2>&1))"

[[ -d .venv ]] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip -q install --upgrade pip

[[ -d sapiens2 ]] || { echo "▶ cloning facebookresearch/sapiens2…"; git clone -q --depth 1 https://github.com/facebookresearch/sapiens2.git; }

echo "▶ installing toolchain (torch pinned LAST so nothing upgrades it)…"
python -m pip -q install coremltools==8.3 huggingface_hub safetensors pillow
python -m pip -q install -e ./sapiens2
python -m pip -q install --force-reinstall torch==2.5.0 torchvision==0.20.0

OUT="$ROOT/weights/SapiensNormal.mlpackage"
python convert_sapiens2_normal.py --size "$SIZE" --out "$OUT"

echo
echo "▶ result:"
du -sh "$OUT"
echo
echo "Next (back in pure-Swift land):"
echo "  swift build -c release"
echo '  BIN=$(swift build -c release --show-bin-path); cp -f weights/mlx.metallib "$BIN/"'
echo "  \"\$BIN/LiToSmoke\" normals weights TMP/quality30.rmbg.png TMP/normals_vis.png"
echo "  ./run.sh sculpt testset/17.jpg out/17 25 3.0 7 3   # check the [refine] Σ|corr| line"
