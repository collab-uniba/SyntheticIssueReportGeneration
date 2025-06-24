import argparse
import json
import yaml
from ollama import chat
from pydantic import BaseModel

class User(BaseModel):
    text: str

parser = argparse.ArgumentParser(description="Zero-shot generation with Ollama")
parser.add_argument("--emotion", default="positive", choices=["positive", "neutral", "negative"], help="Target emotion to inject into prompts")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--num_predict", type=int, default=500, help="Maximum number of tokens to predict")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value")
parser.add_argument("--repeat_penalty", type=float, default=1.1, help="Penalty for repeated tokens")
parser.add_argument("--generations", type=int, default=3, help="Number of generations to produce")
parser.add_argument("--model", type=str, default="llama3.2:1b", help="Model name to use with Ollama (e.g., llama3.2:1b)")
args = parser.parse_args()

with open("prompts.yaml", "r", encoding="utf-8") as f:
    prompt_data = yaml.safe_load(f)
    messages = prompt_data.get("messages", [])
    for msg in messages:
        if "content" in msg:
            msg["content"] = msg["content"].replace("{{emotion}}", args.emotion)

system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")
user_prompt = next((m["content"] for m in messages if m.get("role") == "user"), "")

all_generations = []
print(f"Inizio la generazione di {args.generations} risposte con emozione '{args.emotion}' usando il modello '{args.model}'...")

for i in range(1, args.generations + 1):
    print(f"\n--- Generazione {i} ---")
    try:
        response = chat(
            model=args.model,
            messages=messages,
            format=User.model_json_schema(),
            options={
                "temperature": args.temperature,
                "num_predict": args.num_predict,
                "top_p": args.top_p,
                "repeat_penalty": args.repeat_penalty,
            }
        )
        raw = response.message.content
        try:
            comment_obj = User.model_validate_json(raw)
            print(f"Parsed JSON object: {comment_obj}")
            all_generations.append({
                "emotion": args.emotion,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_text": comment_obj.text
            })
        except Exception as e:
            print(f"Errore JSON parsing alla generazione {i}: {e}")
            all_generations.append({
                "emotion": args.emotion,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "error": "JSON parsing failed",
                "raw_text": raw
            })
    except Exception as e:
        print(f"Errore durante la generazione {i}: {e}")
        all_generations.append({
            "emotion": args.emotion,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "error": str(e)
        })

print("\n--- Tutte le Generazioni Completate ---")
print(json.dumps(all_generations, indent=2, ensure_ascii=False))

filename = f"zero_shot_generation_{args.model.replace(':', '_')}_{args.emotion}.json"
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_generations, f, indent=4, ensure_ascii=False)

print(f"\n✅ Salvate {len(all_generations)} generazioni in '{filename}'")
