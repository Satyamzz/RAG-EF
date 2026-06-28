import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv(dotenv_path=".env")

# --- Pinecone setup ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

index_name = "paulgrahamessay"

# Delete the index if it exists
if index_name in pc.list_indexes().names():
    print(f"Deleting index '{index_name}'...")
    pc.delete_index(index_name)
    print(f"✓ Index '{index_name}' successfully deleted!")
else:
    print(f"Index '{index_name}' does not exist.")

print("\nPinecone workspace cleared!")
