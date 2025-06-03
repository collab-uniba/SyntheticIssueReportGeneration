#!/bin/bash

set -e


if [ "$#" -ne 2 ]; then
  echo "Uso: ./run_training.sh <train_csv_path> <test_csv_path>"
  exit 1
fi

TRAIN_CSV=$1
TEST_CSV=$2

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements_grafico.txt

python3 main.py "$TRAIN_CSV" "$TEST_CSV"

RESULTS_DIR="results"
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Errore: cartella '$RESULTS_DIR' non trovata. Lo script Python potrebbe non essere andato a buon fine."
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="output_${TIMESTAMP}.zip"
zip -r "$ZIP_NAME" "$RESULTS_DIR"

echo "✅ Fatto. File salvato in: $ZIP_NAME"
