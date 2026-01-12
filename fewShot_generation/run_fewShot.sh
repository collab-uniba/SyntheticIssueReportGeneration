#!/usr/bin/env bash
set -e
set -o pipefail

VENV_DIR="venv_fewShot"

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
$PYTHON_CMD -m pip install -r requirements_fewShot.txt

# Verifica se Ollama è installato
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama già installato."
else
    echo "Ollama non trovato."

    case "$OSTYPE" in
        linux*|darwin*)
            echo "Installazione Ollama su Linux/macOS..."
            if ! command -v curl >/dev/null 2>&1; then
                echo "curl non trovato. Installalo prima di procedere."
                exit 1
            fi
            curl -fsSL https://ollama.com/install.sh | sh
            echo "Ollama installato. Rilancia lo script."
            exit 0
            ;;
        msys*|cygwin*|win32*|win64*)
            echo "Windows rilevato. Scarica e installa Ollama manualmente:"
            echo "   https://ollama.com/download/OllamaSetup.exe"
            echo "Dopo l'installazione, rilancia questo script."
            exit 1
            ;;
    esac
fi

# Estrai modello (default: llama3.2:1b)
MODEL_NAME="llama3.2:1b"
next_is_model=false

for arg in "$@"; do
    if [ "$next_is_model" = true ]; then
        MODEL_NAME="$arg"
        break
    fi
    if [ "$arg" = "--model" ]; then
        next_is_model=true
    fi
done

# Rimuovi i ":" per nome file valido
MODEL_SAFE_NAME=$(echo "$MODEL_NAME" | tr ':' '_')

echo "Download del modello '$MODEL_NAME'..."
ollama pull "$MODEL_NAME"

# Crea cartella output
OUTPUT_DIR="fewShot_outputs"
mkdir -p "$OUTPUT_DIR"

# Emozioni da ciclare
emotions=("positive" "negative" "neutral")

for emotion in "${emotions[@]}"; do
    echo "Esecuzione fewShot_generation.py per emotion: $emotion"
    $PYTHON_CMD fewShot_generation.py "$@" --model "$MODEL_NAME" --target_polarity "$emotion" --output_dir "$OUTPUT_DIR"
done

# Crea archivio ZIP
ZIP_FILE="fewShot_outputs.zip"

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
fi