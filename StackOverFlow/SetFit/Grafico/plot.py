import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grafico_outputs")

results = []

# leggi tutti i file metrics
for file in os.listdir(output_dir):
    if file.startswith("metrics_sample_") and file.endswith(".json"):
        path = os.path.join(output_dir, file)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            results.append(data)

# crea dataframe
df_results = pd.DataFrame(results)

# ordina per sample size
df_results = df_results.sort_values("sample_size")

x = df_results["sample_size"].values
y = df_results["f1_score"].values

# trova il knee
kneedle = KneeLocator(x, y, curve="concave", direction="increasing", S=1.0)

knee_x = kneedle.knee
knee_y = kneedle.knee_y

# plot
plt.figure(figsize=(10, 6))
plt.plot(df_results["sample_size"], df_results["f1_score"], marker='o', label="F1-score (macro)")
plt.title("F1-score vs Sample Size")
plt.xlabel("Numero di campioni nel training set")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(df_results["sample_size"])
plt.tight_layout()

if not results:
    raise ValueError("Nessun file metrics trovato in Grafico_outputs")

if knee_x is not None and knee_y is not None:
    plt.axvline(x=knee_x, linestyle='--', color='red', label=f"Knee = {knee_x}")
    plt.scatter(knee_x, knee_y)
    plt.text(knee_x, knee_y, f"Knee: {knee_x}", fontsize=10, ha='right', va='bottom')

plt.legend(loc="lower right")    
print(f"Knee point: {knee_x} with F1-score: {knee_y}")

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")
plt.savefig(plot_path)
plt.show()