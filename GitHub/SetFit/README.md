# SetFit

Questo progetto utilizza SetFit (Sentence Transformer Fine-tuning) per la classificazione del sentiment in testi brevi.
È progettato per essere eseguito facilmente tramite uno script Python (train_setfit.py) e uno script Bash (run_train_setfit.sh).
Supporta anche il training few-shot specificando quanti esempi usare per classe.

## Contenuto del Repository
- train_setfit.py: Script principale per il training e la predizione.
- run_train_setfit.sh: Script shell per gestire l'ambiente virtuale, le dipendenze e l'esecuzione.
- SetFit_results/: cartella che contiene tutte le predizioni e le metriche generate dagli esperimenti.

## Preparazione
Assicurati che i tuoi file train.csv e test.csv abbiano almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_train_setfit.sh
- ./run_train_setfit.sh train_*.csv test_*.csv (opzionale)

## Outputs
- I file .csv e .json vengono salvati in: SetFit_outputs/