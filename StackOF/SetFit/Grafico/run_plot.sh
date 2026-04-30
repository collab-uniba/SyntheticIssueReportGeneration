#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="plot.py"





# -------------------------
# check script
# -------------------------
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# -------------------------
# run
# -------------------------
echo "Genero grafico..."
python "$SCRIPT_NAME"

# -------------------------
# deactivate
# -------------------------
deactivate
echo "Virtual environment disattivato"