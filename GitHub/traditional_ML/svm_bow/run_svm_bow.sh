#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="train_svm_bow.py"

# Controlla che lo script esista
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Errore: file $SCRIPT_NAME non trovato"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"

# Controlla che Python sia disponibile
if ! command -v python3 &> /dev/null; then
    echo "Errore: python3 non trovato"
    exit 1
fi

echo "Avvio di $SCRIPT_NAME..."
echo "Traditional ML - SVM + Bag of Words"

python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE"

echo "Esecuzione completata."