# Zero-Shot Generation con Ollama

Questo progetto esegue generazione zero-shot di testi condizionati da una emozione (positive, neutral, negative) utilizzando Ollama (llama3.2:1b) e Python 3.12.

Il modello riceve solo un prompt che descrive il compito (ad esempio “Scrivi un testo con emozione positiva”) senza esempi concreti.

## Contenuto del repository
- zeroShot_generation.py -> script Python per la generazione zero-shot.
- run_zeroShot.sh -> script Bash per:
  - gestione ambiente virtuale
  - download modello Ollama
  - esecuzione per ciascuna emotion
- prompt.yaml: prompt usati per la generazione.
- zeroShot_outputs/: cartella generata automaticamente con i file .json.

## Prerequisiti
- Ollama installato e funzionante
  - https://ollama.com/download
- Connessione internet

## Preparazione
Assicurati che sia presente:
- prompt.yaml

## Esecuzione
'''bash

- chmod +x run_zeroShot.sh
- ./run_zeroShot.sh (oppure bash run_zeroShot.sh)

## Output
- I file .json vengono salvati automaticamente in: zeroShot_outputs/
- Al termine viene creato: zeroShot_outputs.zip
