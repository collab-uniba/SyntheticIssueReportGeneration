import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from setfit import SetFitModel

print("=== GPU INFO ===")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. TEST ROBERTA
print("\n=== TEST ROBERTA ===")

model_name = "roberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.to(device)

inputs = tokenizer("This is a test sentence", return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print("RoBERTa OK on:", next(model.parameters()).device)

# 2. TEST SETFIT
print("\n=== TEST SETFIT ===")

setfit_model = SetFitModel.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2"
)

embedding = setfit_model.model_body.encode(
    ["This is a test sentence"],
    convert_to_tensor=True,
    device=device
)

print("SetFit embedding device:", embedding.device)

print("\nALL TESTS COMPLETED")