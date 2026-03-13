# SyntheticIssueReportGeneration

Questo repository raccoglie diversi esperimenti e pipeline per classificazione del sentiment e generazione di testi sintetici, utilizzando approcci:
- Few-Shot generation (Ollama)
- Zero-Shot generation (Ollama)
- Fine-tuning (RoBERTa)
- Few-Shot classification (SetFit)

Il progetto è stato sviluppato e testato su Windows utilizzando Git Bash, ma tutto il codice è progettato per essere portabile ed eseguibile anche su Linux e macOS senza modifiche.

## Ambiente di sviluppo utilizzato
- Sistema operativo: Windows 10 / 11
- Shell: Git Bash
- Python: 3.11
- Gestione ambienti: venv
- Script di orchestrazione: Bash (.sh)

Git Bash fornisce un ambiente Unix-like su Windows, permettendo l’uso degli stessi script Bash usati su Linux/macOS.

## Portabilità del codice
Tutti gli script .sh del progetto sono stati scritti seguendo queste regole:
- uso di #!/usr/bin/env bash
- rilevamento automatico di python / python3
- attivazione del virtual environment cross-platform:
  - venv/bin/activate (Linux/macOS)
  - venv/Scripts/activate (Windows + Git Bash)
- creazione automatica delle cartelle di outputs
- gestione sicura degli errori (set -e, set -o pipefail)
- creazione di ZIP degli outputs

Lo stesso script funziona senza modifiche su Windows, Linux e macOS.

## Struttura del repository
Ogni sotto-cartella contiene un progetto indipendente con:
- script Python (.py)
- script Bash (.sh)
- file requirements.txt
- README dedicato

## Prerequisiti
Prerequisiti comuni (tutti i sistemi):
- Python 3.10+
  - https://www.python.org/ (selezionare “Add Python to PATH” durante l’installazione)
- pip
- Connessione internet (per download modelli / dipendenze)

  ### Per Windows (Git Bash)
  Scaricare e installare Git Bash
  - https://git-scm.com/downloads

  ### Per Linux / macOs
  Non è necessario nulla

## Eseguire gli script
'''bash

- chmod +x run_*.sh
- ./run_*.sh [argomenti se previsti]

## Output e risultati
Ogni progetto:
- crea una cartella *_outputs contenente i risultati
- genera il proprio venv
- genera un archivio .zip con i risultati finali
