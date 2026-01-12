#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_Result"
SCRIPT_NAME="train_model.py"

TEST_FILE="test_stackoverflowOriginali.csv"
OUTPUT_DIR="Predictions"

TRAIN_FILES=(
    "fewShot_combined.csv"
    "zeroShot_combined.csv"
    "fewShot_output.csv"
    "train_stackoverflowOriginale.csv"
    "zeroShot_output.csv"
)

# Pulizia output precedente e creazione nuovo
rm -rf "$OUTPUT_DIR" "$ZIP_NAME"
mkdir -p "$OUTPUT_DIR"

# Alias python3 su Windows Git Bash
if command -v py &>/dev/null; then
    PYTHON_CMD=py
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
$PYTHON_CMD -m pip install -r requirements.txt

# Esecuzione script Python
for TRAIN_FILE in "${TRAIN_FILES[@]}"; do
    echo "🚀 Running training with:"
    echo "    ▶️ Train: $TRAIN_FILE"
    echo "    🧪 Test : $TEST_FILE"

    python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE"

    BASE_NAME=$(basename "$TRAIN_FILE" .csv)
    OUTPUT_FILE="test_predictions_${BASE_NAME}.csv"

    if [ -f "$OUTPUT_FILE" ]; then
        mv "$OUTPUT_FILE" "$OUTPUT_DIR/"
    else
        echo "⚠️ Warning: Expected output $OUTPUT_FILE not found!"
    fi
done

# Crea archivio ZIP
ZIP_FILE="Predictions.zip"

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
