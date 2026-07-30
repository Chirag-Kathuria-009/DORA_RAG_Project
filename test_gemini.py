import google.generativeai as genai
import os
from dotenv import load_dotenv
import cohere


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)
cohere_api_key = os.getenv("COHERE_API_KEY")
#cohere.configure(api_key=cohere_api_key)

co = cohere.Client(cohere_api_key)
model_id = "rerank-english-v3.0"  # Replace with the model you want to check
try:
    # Attempt to get details for this specific model
    model = co.models.get(model=model_id)
    
    print(f"✅ Access Confirmed: {model.name}")
    print(f"   Endpoints: {', '.join(model.endpoints)}")
    if model.is_deprecated:
        print("   ⚠️ Warning: This model is deprecated.")
        
except cohere.errors.NotFoundError:
    print(f"❌ Model '{model_id}' not found or not accessible with this key.")
except cohere.errors.UnauthorizedError:
    print("❌ Invalid API Key.")
except Exception as e:
    print(f"❌ Error: {e}")


'''print("API Key loaded:", api_key[:8], "...")
print()

# Step 1: List all models available to your key
print("=== Models available to your API key ===")
for model in genai.list_models():
    if "embedContent" in model.supported_generation_methods:
        print(f"EMBEDDING: {model.name}")
    if "generateContent" in model.supported_generation_methods:
        print(f"CHAT:      {model.name}")'''