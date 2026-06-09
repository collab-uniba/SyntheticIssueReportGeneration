#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="plot_f1_vs_sample_size_mb.py"

# Controllo degli argomenti
if [ "$#" -lt 2 ]; then
    echo "Usage: ./run_plot_mb.sh path/to/train.csv path/to/test.csv"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# Avvio script
echo "Avvio training..."
python "$SCRIPT_NAME" "$TRAIN_FILE" "$TEST_FILE"

echo "Training completato. Risultati salvati in: $OUTPUT_DIR"

