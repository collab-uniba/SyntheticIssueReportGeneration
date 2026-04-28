#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_Grafico"
SCRIPT_NAME="plot.py"

# -------------------------
# venv
# -------------------------
if [ ! -d "$VENV_DIR" ]; then
    echo "Creo virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "Python attivo: $(which python)"

# -------------------------
# install solo se necessario
# -------------------------
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
echo "Genero grafico..."
python "$SCRIPT_NAME"

# -------------------------
# deactivate
# -------------------------
deactivate
echo "Virtual environment disattivato"