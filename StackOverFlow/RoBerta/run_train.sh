#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="train_and_predict.py"

# Controllo argomenti
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_train.sh path/to/train.csv path/to/test.csv"
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

# Crea cartella output
OUTPUT_DIR="RoBerta_outputs"
mkdir -p "$OUTPUT_DIR"

# Esecuzione script Python
echo "Avvio training e prediction..."
python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" --output_dir "$OUTPUT_DIR"

echo "Training e prediction completati. Risultati salvati in: $OUTPUT_DIR"
