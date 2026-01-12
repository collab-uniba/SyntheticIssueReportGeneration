#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_RoBerta"

SCRIPT_NAME="train_and_predict.py"

# Controllo argomenti
if [ "$#" -ne 2 ]; then
    echo "Usage: ./run_train.sh path/to/train.csv path/to/test.csv"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"

# Alias python3 su Windows Git Bash
if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=python
elif command -v python3 >/dev/null 2>&1; then
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
    source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"
else
    echo "Virtual environment non trovato!"
    exit 1
fi

echo "Virtual environment attivato: $VIRTUAL_ENV"

# Installazione requirements
echo "Installazione requirements..."
$PYTHON_CMD -m pip install --upgrade pip
$PYTHON_CMD -m pip install -r requirements.txt

# Crea cartella output
OUTPUT_DIR="RoBerta_outputs"
mkdir -p "$OUTPUT_DIR"

# Esecuzione script Python
echo "Avvio training e prediction..."
$PYTHON_CMD "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE" --output_dir "$OUTPUT_DIR"

# Crea archivio ZIP
ZIP_FILE="RoBerta_outputs.zip"

if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR")" ]; then
    case "$OSTYPE" in
        linux*|darwin*)
            zip -r "$ZIP_FILE" "$OUTPUT_DIR"
            ;;
        msys*|cygwin*|win32*|win64*)
            WINRAR="/c/Program Files/WinRAR/WinRAR.exe"
            "$WINRAR" a -afzip "$ZIP_FILE" "$OUTPUT_DIR"
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
