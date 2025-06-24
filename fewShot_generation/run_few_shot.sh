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

MODEL_NAME="llama3.2:1b"  # default

for i in "$@"; do
    if [[ $i == --model ]]; then
        next_is_model=true
    elif [[ $next_is_model == true ]]; then
        MODEL_NAME=$i
        break
    fi
done

echo "🤖 Download del modello '$MODEL_NAME'..."
ollama pull "$MODEL_NAME"

echo "🚀 Esecuzione dello script Python..."
python fewShot_generation.py "$@"
