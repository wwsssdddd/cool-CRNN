#!/bin/zsh
set -euo pipefail

PROJECT_ROOT="${0:A:h:h}"
DATA_ROOT="${CRNN_MJSYNTH_ROOT:-$PROJECT_ROOT/data/mjsynth}"
MANIFEST_ROOT="${CRNN_MJSYNTH_MANIFEST_ROOT:-$PROJECT_ROOT/data/manifests}"
OUTPUT_ROOT="${CRNN_MODELS_PATH:-$PROJECT_ROOT/models/mjsynth-original-v2}"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  print -u2 "Python окружения не найден: $PYTHON"
  exit 1
fi
if [[ ! -f "$DATA_ROOT/annotation_train.txt" ]]; then
  print -u2 "Не найден $DATA_ROOT/annotation_train.txt"
  exit 1
fi
if [[ ! -f "$DATA_ROOT/annotation_val.txt" ]]; then
  print -u2 "Не найден $DATA_ROOT/annotation_val.txt"
  exit 1
fi

mkdir -p "$MANIFEST_ROOT" "$OUTPUT_ROOT"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$OUTPUT_ROOT/.matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -f "$MANIFEST_ROOT/mjsynth_train.tsv" ]]; then
  "$PYTHON" -m src.prepare_manifest \
    --root "$DATA_ROOT" \
    --annotations "$DATA_ROOT/annotation_train.txt" \
    --format mjsynth \
    --train-output "$MANIFEST_ROOT/mjsynth_train.tsv" \
    --val-fraction 0 \
    --stream \
    --skip-file-check
fi

if [[ ! -f "$MANIFEST_ROOT/mjsynth_val.tsv" ]]; then
  "$PYTHON" -m src.prepare_manifest \
    --root "$DATA_ROOT" \
    --annotations "$DATA_ROOT/annotation_val.txt" \
    --format mjsynth \
    --train-output "$MANIFEST_ROOT/mjsynth_val.tsv" \
    --val-fraction 0 \
    --stream \
    --skip-file-check
fi

export CRNN_DATA_PATH="$DATA_ROOT"
export CRNN_TRAIN_MANIFEST="$MANIFEST_ROOT/mjsynth_train.tsv"
export CRNN_VAL_MANIFEST="$MANIFEST_ROOT/mjsynth_val.tsv"
export CRNN_DEVICE="${CRNN_DEVICE:-mps}"
export CRNN_MODELS_PATH="$OUTPUT_ROOT"
# Официальный bgshih/crnn обучается по числу minibatch-итераций.
export CRNN_MAX_ITERATIONS="${CRNN_MAX_ITERATIONS:-2000000}"
export CRNN_DISPLAY_INTERVAL="${CRNN_DISPLAY_INTERVAL:-100}"
export CRNN_VALIDATION_INTERVAL="${CRNN_VALIDATION_INTERVAL:-1000}"
export CRNN_VALIDATION_SAMPLES="${CRNN_VALIDATION_SAMPLES:-5000}"
export CRNN_SNAPSHOT_INTERVAL="${CRNN_SNAPSHOT_INTERVAL:-10000}"
export CRNN_BATCH_SIZE="${CRNN_BATCH_SIZE:-64}"
export CRNN_VAL_BATCH_SIZE="${CRNN_VAL_BATCH_SIZE:-256}"
export CRNN_NUM_WORKERS="${CRNN_NUM_WORKERS:-0}"
export CRNN_LR="${CRNN_LR:-1.0}"
export CRNN_TRAIN_IMAGE_WIDTH="${CRNN_TRAIN_IMAGE_WIDTH:-100}"
export CRNN_GRAD_CLIP_NORM="${CRNN_GRAD_CLIP_NORM:-0}"
export CRNN_SEED="${CRNN_SEED:-0}"
export MPLBACKEND="Agg"

if [[ -z "${CRNN_RESUME:-}" && -f "$OUTPUT_ROOT/last.pt" ]]; then
  export CRNN_RESUME="$OUTPUT_ROOT/last.pt"
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -u -m src.pipeline
