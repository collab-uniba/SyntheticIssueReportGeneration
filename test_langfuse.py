import os

print("PUBLIC:", os.getenv("LANGFUSE_PUBLIC_KEY"))
print("SECRET:", os.getenv("LANGFUSE_SECRET_KEY"))
print("HOST:", os.getenv("LANGFUSE_HOST"))

print("Connessione riuscita")