#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_SetFit"

SCRIPT_NAME="train_model.py"

# Controllo degli argomenti
if [ "$#" -lt 1 ]; then
    echo "Usage: ./train_and_run.sh path/to/train.csv [path/to/test.csv] [num_samples] [split_ratio]"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"
NUM_SAMPLES="${3:-20}"      # default = 20
SPLIT_RATIO="${4:-0.3}"     # default = 0.3

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# Crea cartella output
OUTPUT_DIR="SetFit_outputs"
mkdir -p "$OUTPUT_DIR"

# Esecuzione script Python
if [ -n "$TEST_FILE" ]; then
    echo "Running training con test set..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO" --output_dir "$OUTPUT_DIR"
else
    echo "Running training senza test set (split automatico)..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO" --output_dir "$OUTPUT_DIR"
fi

echo "Training completato. Risultati salvati in: $OUTPUT_DIR"

