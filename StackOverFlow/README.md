# SyntheticIssueReportGeneration - StackOverFlow dataset

Questo repository raccoglie diversi esperimenti e pipeline per classificazione del sentiment e generazione di testi sintetici, utilizzando approcci:
- Few-Shot generation (Ollama)
- Zero-Shot generation (Ollama)
- Fine-tuning (RoBERTa)
- Few-Shot classification (SetFit)

## Ambiente di sviluppo utilizzato
- Python: 3.12
- Gestione ambienti: venv_tesi
- Script di orchestrazione: Bash (.sh)

## Portabilità del codice
Tutti gli script .sh del progetto sono stati scritti seguendo queste regole:
- uso di #!/usr/bin/env bash
- check del virtual environment
- creazione automatica delle cartelle di outputs
- gestione sicura degli errori (set -e, set -o pipefail)

## Struttura del repository
Ogni sotto-cartella contiene un progetto indipendente con:
- script Python (.py)
- script Bash (.sh)
- README dedicato

## Prerequisiti
- Python 3.10+
  - https://www.python.org/ (selezionare “Add Python to PATH” durante l’installazione)
- pip
- Connessione internet (per download modelli / dipendenze)
- GPU con sufficiente potenza di calcolo

## Eseguire gli script
'''bash

- chmod +x run_*.sh
- ./run_*.sh [argomenti se previsti]

## Output e risultati
Ogni progetto crea una cartella *_outputs contenente i risultati