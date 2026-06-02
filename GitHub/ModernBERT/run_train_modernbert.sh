#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="train_modernbert.py"

# Controllo degli argomenti
if [ "$#" -lt 1 ]; then
    echo "Usage: ./run_train_modernbert.sh path/to/train.csv [path/to/test.csv]"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="${2:-}"      # default = empty

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# Crea cartella output
OUTPUT_DIR="ModernBERT_outputs"
mkdir -p "$OUTPUT_DIR"

# Esecuzione script Python
if [ -n "$TEST_FILE" ]; then
    echo "Running training con test set..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" --output_dir "$OUTPUT_DIR"
else
    echo "Running training senza test set (split automatico)..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" --output_dir "$OUTPUT_DIR"
fi

echo "Training completato. Risultati salvati in: $OUTPUT_DIR"

