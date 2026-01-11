#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_SetFit"

SCRIPT_NAME="train_model.py"

# Controllo degli argomenti
if [ "$#" -lt 1 ]; then
    echo "Usage: ./train_and_run.sh path/to/train.csv [path/to/test.csv] [num_samples] [split_ratio]"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"
NUM_SAMPLES="${3:-20}"      # default = 20
SPLIT_RATIO="${4:-0.3}"     # default = 0.3

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

# Attivazione virtualenv cross-plattform
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"      # Linux / macOS
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"  # Windows Git Bash
else
    echo "Virtual environment non trovato!"
    exit 1
fi

echo "Virtual environment attivato: $VIRTUAL_ENV"

# Installazione requirements
echo "Installing requirements..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Crea cartella output
OUTPUT_DIR="SetFit_outputs"
mkdir -p "$OUTPUT_DIR"

# Esecuzione script Python
if [ -n "$TEST_FILE" ]; then
    echo "Running training con test set..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO" --output_dir "$OUTPUT_DIR"
else
    echo "Running training senza test set (split automatico)..."
    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -n "$NUM_SAMPLES" -s "$SPLIT_RATIO" --output_dir "$OUTPUT_DIR"
fi

# Crea archivio ZIP
ZIP_FILE="SetFit_outputs.zip"

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
    case "$OSTYPE" in
        linux*|darwin*)
            zip -r "$ZIP_FILE" "$OUTPUT_DIR"
            ;;
        msys*|cygwin*|win32*|win64*)
            SEVEN_ZIP="/c/Program Files/7-Zip/7z.exe"
            "$SEVEN_ZIP" a "$ZIP_FILE" "$OUTPUT_DIR"
            ;;
    esac
    echo "Archivio creato: $ZIP_FILE"
else
    echo "Nessun file da archiviare"
fi

# Disattiva il venv in modo sicuro
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Virtual environment disattivato"
fi
