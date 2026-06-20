#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="train_setfit.py"

# Controllo degli argomenti
if [ "$#" -lt 1 ]; then
    echo "Usage: ./run_train_setfit.sh path/to/train.csv [path/to/test.csv] [num_samples]"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="${2:-}"      # default = empty
NUM_SAMPLES="${3:-100}"

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
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" --num_samples "$NUM_SAMPLES" --output_dir "$OUTPUT_DIR"
else
    echo "Running training senza test set (split automatico)..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" --num_samples "$NUM_SAMPLES" --output_dir "$OUTPUT_DIR"
fi

echo "Training completato. Risultati salvati in: $OUTPUT_DIR"