#!/bin/bash

set -e

if [ "$#" -ne 1 ]; then
  echo "Uso: ./run_new_training.sh <train_csv_path>"
  exit 1
fi

TRAIN_CSV=$1

SCRIPTS_DIR="SetFit/Grafico"

cd "$SCRIPTS_DIR" || { echo "Errore: cartella '$SCRIPTS_DIR' non trovata."; exit 1; }

if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

source venv/bin/activate

pip install --upgrade pip
pip install -r requirements_grafico.txt

cd "../.." || { echo "Errore: cartella radice non trovata."; exit 1; }

python3 SetFit/Grafico/new_plot.py "$TRAIN_CSV"

RESULTS_DIR="SetFit/Grafico/results"
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Errore: cartella '$RESULTS_DIR' non trovata. Lo script Python potrebbe non essere andato a buon fine."
  exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="output_newplot_${TIMESTAMP}.zip"
zip -r "$ZIP_NAME" "$RESULTS_DIR"

echo "✅ Fatto. File salvato in: $ZIP_NAME"
