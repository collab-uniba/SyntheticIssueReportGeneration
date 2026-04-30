#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="plot_f1_vs_sample_size.py"

if [ "$#" -ne 2 ]; then
  echo "Uso: ./run_train_final.sh <train_csv> <test_csv>"
  exit 1
fi

TRAIN_CSV=$1
TEST_CSV=$2

# -------------------------
# check script
# -------------------------
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# -------------------------
# check python (usa venv attivo)
# -------------------------
echo "Python attivo: $(which python)"

# -------------------------
# run
# -------------------------
echo "Avvio training..."
python "$SCRIPT_NAME" "$TRAIN_CSV" "$TEST_CSV"