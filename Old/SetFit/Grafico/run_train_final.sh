#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_Grafico"

SCRIPT_NAME="plot_f1_vs_sample_size.py"
 
 # Controllo degli argomenti
if [ "$#" -ne 2 ]; then
  echo "Uso: ./run_train_final.sh <train_csv_path> <test_csv_path>"
  exit 1
fi
 
TRAIN_CSV=$1
TEST_CSV=$2

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
$PYTHON_CMD -m pip install -r requirements_grafico.txt
 
# Esecuzione script Python
echo "Avvio Grafico..."
python "$SCRIPT_NAME" "$TRAIN_CSV" "$TEST_CSV"
 
# Creazione ZIP
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ZIP_NAME="SetFit/Grafico/Grafico_outputs_${TIMESTAMP}.zip"

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

echo "Fatto. File salvato in: $ZIP_NAME"

# Disattiva il venv in modo sicuro
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
    echo "Virtual environment disattivato"
fi
