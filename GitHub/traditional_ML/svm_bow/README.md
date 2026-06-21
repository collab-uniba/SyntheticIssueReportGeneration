# SVM + Bag of Words

Questo progetto utilizza il modello Support Vector Machine (SVM) con features Bag of Words e n-grams per la classificazione del sentiment in testi brevi.
È progettato per essere eseguito facilmente tramite uno script Python (train_svm_bow.py) e uno script Bash (run_svm_bow.sh).

## Contenuto del Repository
- train_svm_bow.py: Script principale per il training, la valutazione e la predizione.
- run_svm_bow.sh: Script per gestire l'esecuzione.
- svm_bow_results/: cartella che contiene tutte le predizioni e le metriche generate dagli esperimenti.

## Preparazione
Assicurati che i tuoi file train.csv e test.csv abbiano almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_svm_bow.sh
- ./run_svm_bow.sh train_*.csv test_*.csv

## Outputs
- I file .csv e .json vengono salvati in: svm_bow_outputs/
