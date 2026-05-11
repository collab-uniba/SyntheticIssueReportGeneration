#!/usr/bin/env bash
set -e
set -o pipefail

SCRIPT_NAME="fewShot_generation.py"

# check script
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "Script $SCRIPT_NAME non trovato"
    exit 1
fi

# check python (usa venv attivo)
echo "Python attivo: $(which python)"

# Verifica se Ollama è installato
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama già installato."
else
    echo "Ollama non trovato."
    echo "Installazione Ollama su Linux/macOS..."

    if ! command -v curl >/dev/null 2>&1; then
        echo "curl non trovato. Installalo prima di procedere."
        exit 1
    fi
    curl -fsSL https://ollama.com/install.sh | sh
    echo "Ollama installato. Rilancia lo script."
    exit 0
fi

# Estrai modello (default: llama3.2:1b)
MODEL_NAME="llama3.2:1b"

ARGS=()
next_is_model=false

for arg in "$@"; do
    if [ "$next_is_model" = true ]; then
        MODEL_NAME="$arg"
        next_is_model=false
        continue
    fi

    if [ "$arg" = "--model" ]; then
        next_is_model=true
        continue
    fi

    ARGS+=("$arg")
done

echo "Modello selezionato: $MODEL_NAME"

# Download del modello
echo "Download del modello '$MODEL_NAME'..."
ollama pull "$MODEL_NAME"

# Crea cartella output
OUTPUT_DIR="fewShot_outputs"
mkdir -p "$OUTPUT_DIR"

# Emozioni da ciclare
emotions=("positive" "negative" "neutral")

for emotion in "${emotions[@]}"; do
    echo "Esecuzione fewShot_generation.py per emotion: $emotion"
    python "$SCRIPT_NAME" "${ARGS[@]}" --model "$MODEL_NAME" --target_polarity "$emotion" --output_dir "$OUTPUT_DIR"
done