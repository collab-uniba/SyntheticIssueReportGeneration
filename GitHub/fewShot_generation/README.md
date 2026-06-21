# Few-Shot Generation con Ollama

Questo progetto esegue generazione di testi in modalità few-shot utilizzando modelli LLM locali tramite Ollama (llama3.1:8b) e Python 3.12.

Viene preso un piccolo numero di esempi reali dal dataset originale (train_StackOverFlow.csv) per guidare la generazione.

Il modello genera nuovi testi seguendo lo stile e la distribuzione degli esempi forniti.

L’esecuzione è automatizzata tramite uno script Python (fewShot_generation.py) e uno script Bash (run_fewShot.sh).

## Contenuto del repository
- fewShot_generation.py -> script Python per la generazione few-shot.
- run_fewShot.sh -> script Bash per:
  - gestione ambiente virtuale
  - download modello Ollama
  - esecuzione per ciascuna emotion
- prompt.yaml: prompt usato per la generazione.
- fewShot_outputs/: cartella generata automaticamente con i file .json.

## Prerequisiti
- Ollama installato e funzionante
  - https://ollama.com/download
- Connessione internet

## Preparazione
Assicurati che siano presenti:
- train_github.csv nella cartella datasets
- prompt.yaml

## Esecuzione
'''bash

- chmod +x run_fewShot.sh
- ./run_fewShot.sh (oppure bash run_fewShot.sh)

## Output
- I file .json vengono salvati automaticamente in: fewShot_outputs/
