#!/bin/bash

VENV_DIR="venv_fewshot"

if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creazione virtual environment in '$VENV_DIR'..."
    python3 -m venv "$VENV_DIR"
fi

echo "🔧 Attivazione virtual environment..."
source "$VENV_DIR/bin/activate"

echo "📥 Installazione requirements..."
pip install --upgrade pip
pip install -r requirements_fewShot.txt

# Verifica se ollama è installato
if ! command -v ollama &> /dev/null; then
    echo "📦 Ollama non trovato, installazione in corso..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "✅ Ollama già installato."
fi

MODEL_NAME="llama3.2:1b"  # default
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

echo "🤖 Download del modello '$MODEL_NAME'..."
ollama pull "$MODEL_NAME"

echo "🚀 Esecuzione dello script Python..."
python fewShot_generation.py "$@"
