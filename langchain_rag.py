import os
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

load_dotenv(dotenv_path=".env")

# --- Config ---
INDEX_NAME = "paulgrahamessay"
EMBEDDING_MODEL = "models/gemini-embedding-001"
OUTPUT_DIMENSIONALITY = 1024
LLM_MODEL = "gemini-2.0-flash"  # change to "gemini-3-flash-preview" if needed
TOP_K = 1

# --- Init embeddings + vector store ---
embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    output_dimensionality=OUTPUT_DIMENSIONALITY
)

vectorstore = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

# --- Init Gemini client (direct SDK) ---
gemini_client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# --- Prompt template ---
PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:"""

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

def query(question: str, show_chunks: bool = False) -> str:
    # Retrieve relevant chunks
    docs = retriever.invoke(question)

    if show_chunks:
        print(f"\n── Retrieved {len(docs)} chunks ──")
        for i, doc in enumerate(docs):
            print(f"\n[Chunk {i+1}] index={doc.metadata.get('chunk_index', '?')}")
            print(doc.page_content[:300] + ("..." if len(doc.page_content) > 300 else ""))
        print("\n── Generating answer ──\n")

    context = format_docs(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    # Generate using direct Gemini SDK
    response = gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )

    return response.text

# --- CLI ---
def main():
    print("\n╔══════════════════════════════════════╗")
    print("║   Paul Graham Essay RAG — CLI Chat   ║")
    print("╚══════════════════════════════════════╝")
    print("Commands:  'chunks' = toggle chunk display | 'exit' = quit\n")

    show_chunks = False

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break

        if question.lower() == "chunks":
            show_chunks = not show_chunks
            print(f"[Chunk display: {'ON' if show_chunks else 'OFF'}]\n")
            continue

        try:
            answer = query(question, show_chunks=show_chunks)
            print(f"\nAssistant: {answer}\n")
        except Exception as e:
            print(f"\n[Error]: {e}\n")

if __name__ == "__main__":
    main()