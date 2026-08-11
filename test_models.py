import os
import requests
from dotenv import load_dotenv

# 1. Set your API Key (replace with your key or set via environment variable)
# os.environ["GROQ_API_KEY"] = "your_actual_api_key_here"
load_dotenv()
api_key =  os.getenv("GROQ_API_KEY")



if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# 2. Define headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# 3. Make the request
url = "https://api.groq.com/openai/v1/models"
response = requests.get(url, headers=headers)

# Check for errors
response.raise_for_status()

# 4. Parse JSON and extract only IDs
data = response.json()
model_ids = [model["id"] for model in data["data"]]

# 5. Print results
for model_id in model_ids:
    print(model_id)