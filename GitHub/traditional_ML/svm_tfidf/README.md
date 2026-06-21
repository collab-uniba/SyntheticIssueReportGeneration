# SVM + TF-IDF

Questo progetto utilizza il modello Support Vector Machine (SVM) con features TF-IDF e n-grams per la classificazione del sentiment in testi brevi.
È progettato per essere eseguito facilmente tramite uno script Python (train_svm_tfidf.py) e uno script Bash (run_svm_tfidf.sh).

## Contenuto del Repository
- train_svm_tfidf.py: Script principale per il training, la valutazione e la predizione.
- run_svm_tfidf.sh: Script per gestire l'esecuzione.
- svm_tfidf_results/: cartella che contiene tutte le predizioni e le metriche generate dagli esperimenti.

## Preparazione
Assicurati che i tuoi file train.csv e test.csv abbiano almeno le colonne:

- ID
- Polarity (valori: positive, negative, neutral)
- Text

## Esecuzione
'''bash

- chmod +x run_svm_tfidf.sh
- ./run_svm_tfidf.sh train_*.csv test_*.csv

## Outputs
- I file .csv e .json vengono salvati in: svm_tfidf_outputs/
