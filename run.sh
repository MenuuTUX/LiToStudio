#!/usr/bin/env bash
# LiTo Studio launcher — builds the native engine and runs it with everything wired up.
#
#   ./run.sh            build (release) + launch the LiTo Studio app
#   ./run.sh debug      build (debug)   + launch the app
#   ./run.sh smoke      MLX + weights sanity check (no UI)
#   ./run.sh engine IMG run the full image→splat pipeline on IMG, writing a .ply
#   ./run.sh sculpt IMG image → splat → Sapiens-refined mesh (the quality pipeline)
#
# Why this script exists: MLX-Swift loads its Metal shader library by looking for
# `mlx.metallib` sitting *next to the executable*. SwiftPM doesn't produce that
# file, so we copy weights/mlx.metallib into the build dir after every build.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="release"
case "${1:-}" in
  debug)  CONFIG="debug";  shift ;;
  smoke|engine|sculpt) ;;     # handled below, build debug for speed
esac

# `smoke`/`engine` are dev harnesses — build them debug so iteration is fast.
# `sculpt` stays release: marching cubes + the Poisson solve are pure-Swift hot loops.
SUB="${1:-app}"
if [[ "$SUB" == "smoke" || "$SUB" == "engine" ]]; then CONFIG="debug"; fi

echo "▶ Building ($CONFIG)…"
swift build -c "$CONFIG"
BIN="$(swift build -c "$CONFIG" --show-bin-path)"

# Colocate the MLX Metal library with the binaries so the GPU backend loads.
# Missing on a fresh clone is fine for the app: first-run setup downloads all
# models (incl. the metallib) and the engine colocates it itself.
if [[ -f weights/mlx.metallib ]]; then
  cp -f weights/mlx.metallib "$BIN/mlx.metallib"
elif [[ "$SUB" == "app" ]]; then
  echo "▶ No weights yet — the app's first-run setup will download everything."
else
  echo "✗ weights/mlx.metallib is missing — run the app once (./run.sh) so first-run setup can fetch the models." >&2
  exit 1
fi

# MetalSplatter loads its shaders from a compiled default.metallib inside its
# resource bundle. Xcode's build system compiles the package's .metal files into
# one automatically, but `swift build` just copies the sources — and without the
# metallib, SplatRenderer.init fatalErrors the moment the viewer loads a splat.
MS_BUNDLE="$BIN/MetalSplatter_MetalSplatter.bundle"
if [[ -d "$MS_BUNDLE" ]]; then
  for src in "$MS_BUNDLE"/*.metal; do
    if [[ ! -f "$MS_BUNDLE/default.metallib" || "$src" -nt "$MS_BUNDLE/default.metallib" ]]; then
      echo "▶ Compiling MetalSplatter shaders…"
      (cd "$MS_BUNDLE" && xcrun -sdk macosx metal -O2 -o default.metallib ./*.metal)
      break
    fi
  done
fi

# Point the engine at this checkout's weights/ regardless of where the binary lives.
export LITO_WEIGHTS_DIR="$PWD/weights"

case "$SUB" in
  smoke)
    exec "$BIN/LiToSmoke" smoke weights/lito.safetensors ;;
  engine)
    IMG="${2:?usage: ./run.sh engine <image.png> [out.ply] [steps] [cfg] [seed] [bestOf]}"
    OUT="${3:-out.ply}"; STEPS="${4:-30}"; CFG="${5:-3.0}"; SEED="${6:-}"; BESTOF="${7:-}"
    exec "$BIN/LiToSmoke" engine weights "$IMG" "$OUT" "$STEPS" "$CFG" ${SEED:+"$SEED"} ${BESTOF:+"$BESTOF"} ;;
  sculpt)
    IMG="${2:?usage: ./run.sh sculpt <image.png> [out_base] [steps] [cfg] [seed] [bestOf] [meshRes]}"
    OUT="${3:-sculpt_out}"; STEPS="${4:-25}"; CFG="${5:-3.0}"; SEED="${6:-7}"; BESTOF="${7:-3}"; MESHRES="${8:-256}"
    exec "$BIN/LiToSmoke" sculpt weights "$IMG" "$OUT" "$STEPS" "$CFG" "$SEED" "$BESTOF" "$MESHRES" ;;
  *)
    echo "▶ Launching LiTo Studio…"
    exec "$BIN/LiToStudio" ;;
esac
