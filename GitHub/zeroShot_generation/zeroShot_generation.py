import argparse
import json
import yaml
import os
import torch
from langfuse import Langfuse
from ollama import chat
from pydantic import BaseModel

# Verifica se CUDA è disponibile e stampa informazioni sulla GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device in uso:", device)

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST")
)

class User(BaseModel):
    text: str

parser = argparse.ArgumentParser(description="Zero-shot generation with Ollama")
parser.add_argument("--emotion", default="positive", choices=["positive", "neutral", "negative"], help="Target emotion to inject into prompts")
parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
parser.add_argument("--num_predict", type=int, default=500, help="Maximum number of tokens to predict")
parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling value")
parser.add_argument("--repeat_penalty", type=float, default=1.1, help="Penalty for repeated tokens")
parser.add_argument("--generations", type=int, default=50, help="Number of generations to produce")
parser.add_argument("--prompt_version", type=str, default="balanced", help="Version of the prompt to use (for tracking in Langfuse)")
parser.add_argument("--model", type=str, default="llama3.2:1b", help="Model name to use with Ollama (e.g., llama3.2:1b)")
parser.add_argument("--output_dir", type=str, default=".", help="Directory dove salvare i file JSON")
args = parser.parse_args()

with open("prompt.yaml", "r", encoding="utf-8") as f:
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
    
    trace = langfuse.trace(
        name="zeroshot_generation",
        metadata={
            "prompt_version": args.prompt_version,
            "generation": i,
            "model": args.model,
            "emotion": args.emotion,
            "temperature": args.temperature,
            "num_predict": args.num_predict,
            "top_p": args.top_p,
            "repeat_penalty": args.repeat_penalty,
            "setup": "zero_shot"
        }
    )
    
    print(f"\n--- Generazione {i} ---")
    try: 
        generation = trace.generation(
            name="ollama_call",
            model=args.model,
            input=messages,
            metadata={
                "emotion": args.emotion
            }
        )
        
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
        
        generation.update(
            output=raw
        )
        
        if raw is None or raw.strip() == "":
            raise ValueError("Ollama ha restituito output vuoto")
        try:
            comment_obj = User.model_validate_json(raw)
            
            # Output valido
            trace.score(name="valid_json", value=1)
            # Generazione riuscita indipendentemente dalla validità del contenuto
            trace.score(name="generation_success", value=1)
            
            print(f"Parsed JSON object: {comment_obj}")
            all_generations.append({
                "emotion": args.emotion,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "generated_text": comment_obj.text
            })
        except Exception as e:
            print(f"Errore JSON parsing alla generazione {i}: {e}")
            
            trace.event(
                name="json_parsing_error",
                metadata={
                    "error": str(e),
                    "generation": i
                }
            )
            trace.score(name="valid_json", value=0)
            trace.score(name="generation_success", value=1)
            
            all_generations.append({
                "emotion": args.emotion,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "error": "JSON parsing failed",
                "raw_text": raw
            })
    except Exception as e:
        print(f"Errore durante la generazione {i}: {e}")
        
        trace.event(
            name="generation_error",
            metadata={
                "error": str(e),
                "generation": i
            }
        )
        trace.score(name="valid_json", value=0)
        trace.score(name="generation_success", value=0)
        
        all_generations.append({
            "emotion": args.emotion,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "error": str(e)
        })

print("\n--- Tutte le Generazioni Completate ---")
print(json.dumps(all_generations, indent=2, ensure_ascii=False))

os.makedirs(args.output_dir, exist_ok=True)

filename = os.path.join(args.output_dir, f"zeroShot_generation_{args.model.replace(':', '_')}_{args.emotion}.json")
with open(filename, "w", encoding="utf-8") as f:
    json.dump(all_generations, f, indent=4, ensure_ascii=False)

print(f"\nSalvate {len(all_generations)} generazioni in '{filename}'")

langfuse.flush()