from dotenv import load_dotenv
from pinecone import Pinecone
import os
print("Starting script...")
load_dotenv(dotenv_path=".env")

key = os.environ.get("PINECONE_API_KEY")
print(f"Key found: {bool(key)}")
print(f"Key preview: {repr(key[:12]) if key else 'NONE'}")
print("Listing indexes...")
try:
    pc = Pinecone(api_key=key)
    index_list = pc.list_indexes()
    print(f"Raw API Response: {index_list}")
    
    names = index_list.names()
    print(f"Index Names: {names}")
    
    if not names:
        print("Authentication succeeded, but no indexes were found in this project.")
except Exception as e:
    print(f"An error occurred: {e}")