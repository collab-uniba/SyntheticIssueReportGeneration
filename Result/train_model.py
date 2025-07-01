import argparse
from datasets import ClassLabel, Value, Features
from datasets import load_dataset, DatasetDict
from setfit import SetFitModel, Trainer, TrainingArguments, sample_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import os
import json
 
parser = argparse.ArgumentParser()
 
parser.add_argument("-d", "--train_file", type=str, required=True, help="Percorso al file CSV di training.")
parser.add_argument("-t", "--test_file", type=str, default=None, help="Percorso al file CSV di test (opzionale).")
parser.add_argument("-n", "--num_samples", type=int, default=50, help="Numero di sample per etichetta da usare per il training. 0 = tutto il dataset.")
parser.add_argument("-s", "--split_ratio", type=float, default=0.3, help="Percentuale del dataset di training da usare come test se non è fornito un test set.")
 
args = parser.parse_args()
 
data_files = {"train": args.train_file}
if args.test_file:
    data_files["test"] = args.test_file
 
features = Features({
    "ID": Value("string"),
    "Text": Value("string"),
    "Polarity": Value("string")  # or whatever your labels are
})
 
dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter=";",
    quotechar='"',
    features=features
)
 
# Se non c'è test set, fai uno split del train (caso del dataset di github gold)
if "test" not in dataset:
    df = dataset["train"].to_pandas()
    df_train, df_test = train_test_split(df, test_size=args.split_ratio, stratify=df["Polarity"], random_state=42)
    dataset = DatasetDict({
        "train": dataset["train"].from_pandas(df_train.reset_index(drop=True)),
        "test": dataset["train"].from_pandas(df_test.reset_index(drop=True))
    })
 
# per usare tutto il dataset e non fare un sample passare come parametro n<=0
if args.num_samples > 0:
    train_dataset = sample_dataset(dataset["train"], label_column="Polarity", num_samples=args.num_samples)
else:
    train_dataset = dataset["train"]
 
# Se ci sono esempi di testo vuoto, sostituiscili con " "
train_dataset = train_dataset.map(lambda x: {"Text": x["Text"] if x["Text"] else " "})
 
test_dataset = dataset["test"]
# Apply the same text cleaning to test dataset
test_dataset = test_dataset.map(lambda x: {"Text": x["Text"] if x["Text"] else " "})
 
model = SetFitModel.from_pretrained(
    "all-mpnet-base-v2",
    labels=["negative", "positive", "neutral"],
)
 
training_args = TrainingArguments(
    batch_size=16,
    num_epochs=4,
    eval_strategy="no",
    save_strategy="no",
    load_best_model_at_end=True,
)
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="accuracy",
    column_mapping={"Text": "text", "Polarity": "label"}
)
 
trainer.train()
 
# Use trainer.evaluate() and model.predict() with proper alignment
print("Evaluating model...")
metrics = trainer.evaluate(test_dataset)
print("Trainer evaluation metrics:", metrics)
 
 
# Make predictions maintaining order
predicted_labels = trainer.model.predict(test_dataset["Text"])
 
# Extract true labels in the same order
true_labels = [item["Polarity"] for item in test_dataset]
 
# Debug: Check the types and sample values
print("Sample true labels:", true_labels[:5], "Type:", type(true_labels[0]))
print("Sample predicted labels:", predicted_labels[:5], "Type:", type(predicted_labels[0]))
 
# Convert labels to ensure consistency
# If true_labels are integers, convert predicted_labels to integers
# If true_labels are strings, convert predicted_labels to strings
if isinstance(true_labels[0], int):
    # True labels are integers, convert predictions to integers if they're strings
    if isinstance(predicted_labels[0], str):
        label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
        predicted_labels = [label_to_int.get(pred, pred) for pred in predicted_labels]
else:
    # True labels are strings, convert predictions to strings if they're integers
    if isinstance(predicted_labels[0], (int, float)):
        int_to_label = {0: "negative", 1: "neutral", 2: "positive"}
        predicted_labels = [int_to_label.get(int(pred), str(pred)) for pred in predicted_labels]
 
print("After conversion:")
print("Sample true labels:", true_labels[:5], "Type:", type(true_labels[0]))
print("Sample predicted labels:", predicted_labels[:5], "Type:", type(predicted_labels[0]))
 
# Calculate metrics using aligned data
report = classification_report(true_labels, predicted_labels, output_dict=True)
print("Classification Report:")
print(classification_report(true_labels, predicted_labels))
 
# Create results dataframe with guaranteed alignment
df_results = pd.DataFrame({
    "ID": [item["ID"] for item in test_dataset],
    "Text": [item["Text"] for item in test_dataset],
    "True_Label": true_labels,
    "Prediction": predicted_labels
})
 
dataset_name = os.path.splitext(os.path.basename(args.train_file))[0]
output_filename = f"test_predictions_{dataset_name}.csv"
 
# Save classification report to json
with open(f"classification_report_{dataset_name}.json", "w") as f:
    json.dump(report, f, indent=4)
 
# Save results with both true labels and predictions for verification
df_results.to_csv(output_filename, index=False, encoding="utf-8")
 
print(f"Results saved to {output_filename}")
print(f"Classification report saved to classification_report_{dataset_name}.json")
 
# Verify alignment by recalculating accuracy from saved file
accuracy_from_saved = (df_results["True_Label"] == df_results["Prediction"]).mean()
print(f"Accuracy from trainer evaluation: {report['accuracy']:.4f}")
print(f"Accuracy from saved predictions: {accuracy_from_saved:.4f}")
