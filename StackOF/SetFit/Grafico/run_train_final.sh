#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_Grafico"
SCRIPT_NAME="plot_f1_vs_sample_size.py"

if [ "$#" -ne 2 ]; then
  echo "Uso: ./run_train_final.sh <train_csv> <test_csv>"
  exit 1
fi

TRAIN_CSV=$1
TEST_CSV=$2

# -------------------------
# venv
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Python attivo: $(which python)"

# -------------------------
# upgrade pip
# -------------------------
pip install --upgrade pip

# -------------------------
# requirements (base)
# -------------------------
pip install -r requirements_grafico.txt

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
echo "Avvio training..."
python "$SCRIPT_NAME" "$TRAIN_CSV" "$TEST_CSV"

# -------------------------
# Disattivazione venv
# -------------------------
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Virtual environment disattivato"
fi