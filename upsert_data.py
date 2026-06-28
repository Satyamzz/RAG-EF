import os
import time # Added for rate limiting
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader # Swapped to PDF loader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv(dotenv_path=".env")

# --- Pinecone setup ---
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "blutrain"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# --- Load PDF File ---
loader = PyPDFLoader("arxiv.pdf")
documents = loader.load()

# --- Chunk the Text ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# Add metadata
for i, doc in enumerate(docs):
    doc.metadata = {
        "source": "arxiv.pdf",
        "chunk_index": i
    }

print(f"Loaded and split into {len(docs)} document chunks")

# --- Embeddings Setup ---
# Updated to the modern embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001", 
    output_dimensionality=1024
)  

# Initialize the vector store object just ONCE
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

# --- Embed and upsert in batches with Rate Limit Protection ---
batch_size = 100
for i in range(0, len(docs), batch_size):
    batch = docs[i : i + batch_size]
    
    # Use add_documents instead of from_documents in a loop
    vectorstore.add_documents(batch)
    print(f"Upserted {min(i + batch_size, len(docs))} / {len(docs)}")
    
    # Pause for 3 seconds between batches to respect the Free Tier limits!
    time.sleep(3) 

print(f"Done — {len(docs)} documents in index '{index_name}'")