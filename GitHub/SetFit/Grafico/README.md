# Analisi Few-Shot con SetFit: Performance in base alla quantità di dati

Questo script ha come scopo il valutare le prestazioni di SetFit su diversi sample per verificare dove otteniamo le prestazioni migliori (o per capire da che punto in poi le otteniamo) sulla metrica f1-score.
Lo script genererà alla fine un grafico per rendere visibile il risultato dell'analisi.

## Contenuto del Repository
- plot_f1_vs_sample_size.py
- run_plot.sh
- Grafico_results = cartela contenente il grafico risultante dall'analisi

## Prerequisiti
Per dataset più grandi o per molte ripetizioni è consigliata una GPU con buona potenza di calcolo

## Esecuzione
Per utilizzare lo script si può utilizzare  run_plot.sh per semplificare il run.

Basterà usare il comando chmod +x run_plot.sh la prima volta che si usa lo script e per poi usare ./run_plot.sh train_*.csv test_*.csv per runnarlo senza problemi.
