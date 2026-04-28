#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_data_quality"

DATASET=$1
ORIGINAL=$2

# Alias python3 su Windows Git Bash
if command -v python &>/dev/null; then
    PYTHON_CMD=python
elif command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
else
    echo "Python non trovato. Installa Python e aggiungilo al PATH."
    exit 1
fi

echo "Uso Python: $($PYTHON_CMD --version)"

# Creazione virtualenv se non esiste
if [ ! -d "$VENV_DIR" ]; then
    echo "Creazione virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Attivazione virtualenv cross-platform
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"      # Linux/macOS
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"  # Windows Git Bash
else
    echo "Virtual environment non trovato!"
    exit 1
fi

echo "Virtual environment attivato: $VIRTUAL_ENV"

# Installazione requirements
echo "Installazione requirements..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements_data_quality.txt

echo "Analisi dataset: $DATASET"

python dataset_quality.py \
    --dataset $DATASET --original $ORIGINAL

echo "Analisi completata"