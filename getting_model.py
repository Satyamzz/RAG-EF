import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# Configure with your key
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Call ModelService.ListModels
for model in genai.list_models():
    print(f"Model: {model.name}")
    print(f"Methods: {model.supported_generation_methods}\n")