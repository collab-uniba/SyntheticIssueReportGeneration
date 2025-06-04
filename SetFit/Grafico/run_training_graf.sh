#!/bin/bash
 
set -e
 
 
if [ "$#" -ne 2 ]; then
  echo "Uso: ./run_training.sh <train_csv_path> <test_csv_path>"
  exit 1
fi
 
TRAIN_CSV=$1
TEST_CSV=$2
 
# Create a variable with the path to the scripts dir
SCRIPTS_DIR="SetFit/Grafico"
 
cd "$SCRIPTS_DIR" || { echo "Errore: cartella '$SCRIPTS_DIR' non trovata."; exit 1; }
 
python3 -m venv venv
source venv/bin/activate
 
pip install --upgrade pip
pip install -r requirements_grafico.txt
 
# Move back to the root directory
cd "../.." || { echo "Errore: cartella radice non trovata."; exit 1; }
 
python3 SetFit/Grafico/plot_f1_vs_sample_size.py "$TRAIN_CSV" "$TEST_CSV"
 
RESULTS_DIR="SetFit/Grafico/results"
if [ ! -d "$RESULTS_DIR" ]; then
  echo "Errore: cartella '$RESULTS_DIR' non trovata. Lo script Python potrebbe non essere andato a buon fine."
  exit 1
fi
 
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="output_${TIMESTAMP}.zip"
zip -r "$ZIP_NAME" "$RESULTS_DIR"
 
echo "✅ Fatto. File salvato in: $ZIP_NAME"
