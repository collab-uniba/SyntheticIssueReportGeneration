#!/bin/bash

VENV_DIR="venv"

SCRIPT_NAME="train_and_predict.py"

if [ "$#" -ne 2 ]; then
    echo "❌ Usage: ./run_train.sh path/to/train.csv path/to/test.csv"
    exit 1
fi

TRAIN_FILE="$1"
TEST_FILE="$2"

# Crea l'ambiente virtuale se non esiste
if [ ! -d "$VENV_DIR" ]; then
    echo "🛠 Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Attiva il venv
echo "⚙️ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Installa i pacchetti necessari
echo "📦 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Esegue lo script Python con i file passati
echo "🚀 Running the training and prediction script..."
python "$SCRIPT_NAME" -d "$TRAIN_FILE" -t "$TEST_FILE"

# Disattiva l'ambiente virtuale al termine
deactivate
