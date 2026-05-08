#!/usr/bin/env bash

set -e
set -o pipefail

SCRIPT_NAME="SetFit/Grafico/plot_prel.py"
RESULTS_DIR="SetFit/Grafico/results"

if [ "$#" -ne 1 ]; then
    echo "Uso: ./train_split.sh <train_csv_path>"
    exit 1
fi

TRAIN_CSV=$1

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# avvio script
echo "Avvio plot..."
python "$SCRIPT_NAME" "$TRAIN_CSV"

# check risultati
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Errore: cartella '$RESULTS_DIR' non trovata."
    echo "Lo script Python potrebbe non essere andato a buon fine."
    exit 1
fi

echo "Plot completato con successo."