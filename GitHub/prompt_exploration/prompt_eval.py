import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from langfuse import Langfuse
from langfuse.experiment import LocalExperimentItem

# =========================
# INIT
# =========================
langfuse = Langfuse()
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD DATA
# =========================
def load_dataset(path):
    df = pd.read_csv(path, sep=";", engine="python", on_bad_lines="skip")
    df.columns = ["ID", "Polarity", "Text"]

    df = df.dropna()
    df["Text"] = df["Text"].astype(str)
    return df


real = load_dataset("train_github.csv")

synthetic_a = load_dataset("train_fewShot_creative.csv")
synthetic_b = load_dataset("train_fewShot_balanced.csv")
synthetic_c = load_dataset("train_fewShot_strict.csv")

datasets = {
    "prompt_creative": synthetic_a,
    "prompt_balanced": synthetic_b,
    "prompt_strict": synthetic_c
}

# =========================
# REAL EMBEDDINGS
# =========================
real_emb = model.encode(real["Text"].tolist(), normalize_embeddings=True)
real_centroid = np.mean(real_emb, axis=0)

# =========================
# METRICS
# =========================
def compute_metrics(df):
    texts = df["Text"].tolist()

    emb = model.encode(texts, normalize_embeddings=True)
    centroid = np.mean(emb, axis=0)

    similarity = cosine_similarity([centroid], [real_centroid])[0][0]

    if len(texts) > 1:
        sim_matrix = cosine_similarity(emb)
        upper = sim_matrix[np.triu_indices(len(texts), k=1)]
        diversity = 1 - np.mean(upper)
    else:
        diversity = 0.0

    return {
        "similarity": float(similarity),
        "diversity": float(diversity)
    }

# =========================
# TASK (LANGFUSE)
# =========================
def task(item):
    df = pd.DataFrame(item["input"])
    metrics = compute_metrics(df)

    return {
        "prompt": item["id"],
        "similarity": metrics["similarity"],
        "diversity": metrics["diversity"]
    }

# =========================
# EVALUATORS
# =========================
def eval_similarity(**kwargs):
    return kwargs.get("output", {}).get("similarity", 0.0)

def eval_diversity(**kwargs):
    return kwargs.get("output", {}).get("diversity", 0.0)

# =========================
# EXPERIMENT DATA
# =========================
experiment_data = [
    LocalExperimentItem(
        id=name,
        input=df.to_dict(orient="records")
    )
    for name, df in datasets.items()
]

# =========================
# RUN EXPERIMENT
# =========================
result = langfuse.run_experiment(
    name="prompt_comparison_experiment",
    run_name="run_1",
    data=experiment_data,
    task=task,
    evaluators=[eval_similarity, eval_diversity],
    metadata={
        "model": "llama3.2:1b",
        "task": "synthetic_dataset_evaluation"
    }
)

print(result)

# =========================
# EXPORT RAW RESULTS
# =========================
rows = []
for item in experiment_data:
    df = pd.DataFrame(item["input"])
    metrics = compute_metrics(df)

    rows.append({
        "prompt": item["id"],
        "similarity": metrics["similarity"],
        "diversity": metrics["diversity"]
    })

with open("experiment_raw_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

# =========================
# SUMMARY + RANKINGS (LIGHTWEIGHT)
# =========================
df = pd.DataFrame(rows)

summary = df.groupby("prompt").agg(
    similarity_mean=("similarity", "mean"),
    similarity_std=("similarity", lambda x: float(np.nanstd(x))),
    diversity_mean=("diversity", "mean"),
    diversity_std=("diversity", lambda x: float(np.nanstd(x))),
).reset_index()

rankings = {
    "ranking_similarity": df.groupby("prompt")["similarity"].mean().sort_values(ascending=False).to_dict(),
    "ranking_diversity": df.groupby("prompt")["diversity"].mean().sort_values(ascending=False).to_dict()
}

with open("experiment_rankings.json", "w", encoding="utf-8") as f:
    json.dump(rankings, f, indent=2, ensure_ascii=False)

# =========================
# SINGLE TESI-READY PLOT
# =========================
plot_df = summary.set_index("prompt")

ax = plot_df[["similarity_mean", "diversity_mean"]].plot(kind="bar")
ax.set_title("Prompt comparison: similarity vs diversity")
ax.set_ylabel("score")
ax.set_xlabel("prompt")

plt.xticks(rotation=0)

# =========================
# VALUE LABELS ON TOP
# =========================
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=9
    )

plt.ylim(0, 1)  # utile per leggibilità (similarity/diversity sono 0-1)

plt.tight_layout()
plt.savefig("prompt_comparison_chart.png", dpi=300)
plt.show()

# TABELLA

summary_table = plot_df[["similarity_mean", "diversity_mean"]].copy()

summary_table = summary_table.reset_index()

summary_table.columns = ["prompt", "similarity", "diversity"]

# coversione percentuale
summary_table["similarity"] = (summary_table["similarity"] * 100).round(2)
summary_table["diversity"] = (summary_table["diversity"] * 100).round(2)

fig, ax = plt.subplots(figsize=(8, 2))

ax.axis("off")  # nasconde assi

table = ax.table(
    cellText=summary_table.values,
    colLabels=summary_table.columns,
    cellLoc="center",
    loc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.title("Summary Table (percent values)", pad=20)

plt.tight_layout()
plt.savefig("experiment_summary_table.png", dpi=300, bbox_inches="tight")
plt.show()

print("\n=== SUMMARY TABLE ===")
print(summary_table)

# =========================
# DONE
# =========================
print("\n✔ DONE")
print("Generated files:")
print("- experiment_raw_results.json")
print("- experiment_rankings.json")
print("- experiment_summary_table.png")
print("- prompt_comparison_chart.png")