#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="pred_bartzeroshot.py"

# Controllo argomenti
if [ "$#" -lt 1 ]; then
    echo "Usage: ./run_pred_bartzeroshot.sh path/to/test.csv"
    exit 1
fi

TEST_FILE="$1"

# Controllo script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Errore: file $SCRIPT_NAME non trovato"
    exit 1
fi

# Mostra python attivo (venv)
echo "Python attivo: $(which python)"

echo "Running BART Zero-Shot prediction..."

python "$SCRIPT_NAME" -t "$TEST_FILE"

echo "Prediction completata."