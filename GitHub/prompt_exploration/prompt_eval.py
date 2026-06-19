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


real = load_dataset("github.csv")

synthetic_a = load_dataset("fewShot_creative.csv")
synthetic_b = load_dataset("fewShot_balanced.csv")
synthetic_c = load_dataset("fewShot_strict.csv")

datasets = {
    "prompt_creative": synthetic_a,
    "prompt_balanced": synthetic_b,
    "prompt_strict": synthetic_c
}

# =========================
# REAL EMBEDDINGS
# =========================
real_emb = model.encode(real["Text"].tolist(), normalize_embeddings=True)

# =========================
# METRICS
# =========================
def compute_metrics(real_emb, synth_emb):

    sim_rs = cosine_similarity(synth_emb, real_emb)
    sim_ss = cosine_similarity(synth_emb)

    # similarity: synthetic -> real
    similarity = np.mean(np.max(sim_rs, axis=1))

    # coverage: real -> synthetic (FIX IMPORTANTE)
    coverage = np.mean(np.max(sim_rs, axis=0))

    # diversity: intra-synthetic similarity
    if len(synth_emb) > 1:
        diversity = 1 - np.mean(sim_ss[np.triu_indices(len(synth_emb), k=1)])
    else:
        diversity = 0.0

    # novelty: inverse similarity
    novelty = 1 - similarity

    return {
        "similarity": float(similarity),
        "coverage": float(coverage),
        "diversity": float(diversity),
        "novelty": float(novelty)
    }

# =========================
# TASK (LANGFUSE)
# =========================
def task(item):
    df = pd.DataFrame(item["input"])

    synth_emb = model.encode(df["Text"].tolist(), normalize_embeddings=True)
    metrics = compute_metrics(real_emb, synth_emb)

    return {
        "prompt": item["id"],
        **metrics
    }

# =========================
# EVALUATORS
# =========================
def eval_similarity(**kwargs):
    return kwargs.get("output", {}).get("similarity", 0.0)

def eval_diversity(**kwargs):
    return kwargs.get("output", {}).get("diversity", 0.0)

def eval_coverage(**kwargs):
    return kwargs.get("output", {}).get("coverage", 0.0)

def eval_novelty(**kwargs):
    return kwargs.get("output", {}).get("novelty", 0.0)

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
    evaluators=[eval_similarity, eval_diversity, eval_coverage, eval_novelty],
    metadata={
        "model": "all-MiniLM-L6-v2",
        "task": "synthetic_dataset_evaluation"
    }
)

print(result)

# =========================
# EXPORT RAW RESULTS
# =========================
rows = []

for name, df in datasets.items():
    synth_emb = model.encode(df["Text"].tolist(), normalize_embeddings=True)
    metrics = compute_metrics(real_emb, synth_emb)

    rows.append({
        "prompt": name,
        **metrics
    })

with open("experiment_raw_results.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, indent=2, ensure_ascii=False)

# =========================
# SUMMARY + RANKINGS
# =========================
df = pd.DataFrame(rows)

summary = df.groupby("prompt").agg(
    similarity_mean=("similarity", "mean"),
    diversity_mean=("diversity", "mean"),
    coverage_mean=("coverage", "mean"),
    novelty_mean=("novelty", "mean")
).reset_index()

rankings = {
    "ranking_similarity": df.groupby("prompt")["similarity"].mean().sort_values(ascending=False).to_dict(),
    "ranking_diversity": df.groupby("prompt")["diversity"].mean().sort_values(ascending=False).to_dict(),
    "ranking_coverage": df.groupby("prompt")["coverage"].mean().sort_values(ascending=False).to_dict(),
    "ranking_novelty": df.groupby("prompt")["novelty"].mean().sort_values(ascending=False).to_dict()
}

with open("experiment_rankings.json", "w", encoding="utf-8") as f:
    json.dump(rankings, f, indent=2, ensure_ascii=False)

# =========================
# PLOT (BAR CHART)
# =========================
plot_df = summary.set_index("prompt")

ax = plot_df[["similarity_mean", "diversity_mean"]].plot(kind="bar")
ax.set_title("Prompt comparison: similarity vs diversity")
ax.set_ylabel("score")
ax.set_xlabel("prompt")

plt.xticks(rotation=0)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)

plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("prompt_comparison_chart.png", dpi=300)
plt.show()

# =========================
# TABLE
# =========================
summary_table = plot_df[[
    "similarity_mean",
    "diversity_mean",
    "coverage_mean",
    "novelty_mean"
]].reset_index()

summary_table.columns = ["prompt", "similarity", "diversity", "coverage", "novelty"]

summary_table["similarity"] = (summary_table["similarity"] * 100).round(2).astype(str) + "%"
summary_table["diversity"] = (summary_table["diversity"] * 100).round(2).astype(str) + "%"
summary_table["coverage"] = (summary_table["coverage"] * 100).round(2).astype(str) + "%"
summary_table["novelty"] = (summary_table["novelty"] * 100).round(2).astype(str) + "%"

fig, ax = plt.subplots(figsize=(8, 2))
ax.axis("off")

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
# RADAR CHART
# =========================
metrics = ["similarity_mean", "diversity_mean", "coverage_mean", "novelty_mean"]

radar_df = summary.set_index("prompt")[metrics]

radar_norm = (radar_df - radar_df.min()) / (radar_df.max() - radar_df.min() + 1e-8)

labels = ["Similarity", "Diversity", "Coverage", "Novelty"]
num_vars = len(labels)

angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

for prompt in radar_norm.index:
    values = radar_norm.loc[prompt].tolist()
    values += values[:1]

    ax.plot(angles, values, linewidth=2, label=prompt)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)

ax.set_title("Synthetic Prompt Evaluation Radar", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.savefig("radar_prompt_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# =========================
# DONE
# =========================
print("\n✔ DONE")
print("Generated files:")
print("- experiment_raw_results.json")
print("- experiment_rankings.json")
print("- experiment_summary_table.png")
print("- prompt_comparison_chart.png")
print("- radar_prompt_comparison.png")