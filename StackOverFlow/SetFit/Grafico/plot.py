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

# trova plateau
epsilon = 0.015

df_results["delta_f1"] = df_results["f1_score"].diff()

plateau_points = df_results[df_results["delta_f1"] < epsilon]

plateau_x = None
plateau_y = None

if len(plateau_points) > 0:
    first_plateau = plateau_points.iloc[0]
    plateau_x = int(first_plateau["sample_size"])
    plateau_y = first_plateau["f1_score"]

# plot
xticks = [5, 10, 15, 20, 25, 50, 100, 200, 1000]

plt.figure(figsize=(10, 6))
plt.xscale("log")
plt.plot(df_results["sample_size"], df_results["f1_score"], marker='o', label="F1-score (macro)")
plt.title("F1-score vs Sample Size")
plt.xlabel("Numero di campioni nel training set")
plt.ylabel("F1-score (macro)")
plt.grid(True)
plt.xticks(xticks, labels=[str(x) for x in xticks])

if not results:
    raise ValueError("Nessun file metrics trovato in Grafico_outputs")

# knee
if knee_x is not None and knee_y is not None:
    plt.axvline(x=knee_x, linestyle='--', color='red', label="Knee")

    plt.scatter(knee_x, knee_y, color='red')

    plt.text(knee_x, knee_y, f"Knee: {knee_x}", fontsize=10, ha='right', va='bottom')

# plateau
if plateau_x is not None and plateau_y is not None:
    plt.axvline(x=plateau_x, linestyle='--', color='green', label="Plateau")

    plt.scatter(plateau_x, plateau_y, color='green')

    plt.text(plateau_x, plateau_y, f"Plateau: {plateau_x}", fontsize=10, ha='left', va='top')

plt.legend(loc="lower right")

print(f"Knee point: {knee_x} with F1-score: {knee_y}")
print(f"Plateau point: {plateau_x} with F1-score: {plateau_y}")

plot_path = os.path.join(output_dir, "f1_score_vs_sample_size.png")

plt.tight_layout()
plt.savefig(plot_path)
plt.show()