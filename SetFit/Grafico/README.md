# Analisi Few-Shot con SetFit: Performance in base alla quantità di dati

Questo script ha come scopo il valutare le prestazioni di SetFit su diversi sample per verificare dove otteniamo le prestazioni migliori (o per capire da che punto in poi le otteniamo) sulla metrica f1-score.
Lo script genererà a fine un grafico per rendere visibile il risultato dell'analisi.

## Contenuto del Repository
- f1_score_vs_sample_size.png: Grafico generato automaticamente.
- requirements_grafico.txt: Dipendenze Python per eseguire lo script.
- run_train_final.sh
- plot_f1_vs_sample_size.py

## Prerequisiti
- Python = 3.11
- pip install -r requirements_grafico.txt
- Lo script può essere eseguito su CPU, ma per dataset più grandi o per molte ripetizioni è consigliata una GPU

## Esecuzione
Per utilizzare lo script si può utilizzare  run_train_final.sh per semplificare il run.

Basterà usare il comando chmod +x run_training.sh la prima volta che si usa lo script e per poi usare ./run_train_final.sh path/to/train.csv path/to/test.csv per runnarlo senza problemi. Nel nostro caso scriveremo ./run_train_final.sh ../../train_StackOverFlow.csv ../../test_StackOverFlow.csv
