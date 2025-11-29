"""
Script to setup Chroma vector store with knowledge base.
Creates embeddings for all knowledge base entries.
"""
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chromadb import Client
from chromadb.config import Settings
import google.generativeai as genai
from config import (
    GEMINI_API_KEY,
    KNOWLEDGE_BASE_FILE,
    CHROMA_DB_DIR,
    AgentConfig,
    ModelConfig
)

def setup_chroma_store():
    """Setup Chroma vector store with knowledge base"""
    print("="*60)
    print("Chroma Vector Store Setup")
    print("="*60)

    # Configure Gemini
    genai.configure(api_key=GEMINI_API_KEY)

    # Load knowledge base
    print(f"\n[INFO] Loading knowledge base from {KNOWLEDGE_BASE_FILE}...")
    with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    print(f"[OK] Loaded {len(knowledge_base)} entries")

    # Initialize Chroma client
    print(f"\n[INFO] Initializing Chroma at {CHROMA_DB_DIR}...")
    client = Client(Settings(
        persist_directory=str(CHROMA_DB_DIR),
        anonymized_telemetry=False
    ))

    # Create or get collection
    collection_name = AgentConfig.CHROMA_COLLECTION_NAME
    print(f"[INFO] Creating collection: {collection_name}...")

    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
        print(f"[INFO] Deleted existing collection")
    except:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"description": "Customer support knowledge base"}
    )

    # Prepare documents for embedding
    print(f"\n[INFO] Generating embeddings...")
    documents = []
    metadatas = []
    ids = []

    for entry in knowledge_base:
        # Combine question and answer for better retrieval
        doc_text = f"Q: {entry['question']}\nA: {entry['answer']}"
        documents.append(doc_text)

        metadatas.append({
            "id": str(entry['id']),
            "intent": entry['intent'],
            "category": entry['category'],
            "question": entry['question']
        })

        ids.append(f"kb_{entry['id']}")

    # Generate embeddings using Gemini
    print(f"[INFO] Using Gemini embeddings model...")
    embeddings = []

    # Process in batches to avoid rate limits
    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}...")

        for doc in batch_docs:
            result = genai.embed_content(
                model=ModelConfig.EMBEDDING_MODEL,
                content=doc,
                task_type="retrieval_document"
            )
            embeddings.append(result['embedding'])

    print(f"[OK] Generated {len(embeddings)} embeddings")

    # Add to collection
    print(f"\n[INFO] Adding documents to Chroma collection...")
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    print(f"[OK] Added {len(documents)} documents to collection")

    # Verify
    print(f"\n[INFO] Verifying vector store...")
    count = collection.count()
    print(f"[OK] Collection '{collection_name}' contains {count} documents")

    # Test query
    print(f"\n[INFO] Testing retrieval...")
    test_query = "How do I cancel my order?"
    query_embedding = genai.embed_content(
        model=ModelConfig.EMBEDDING_MODEL,
        content=test_query,
        task_type="retrieval_query"
    )

    results = collection.query(
        query_embeddings=[query_embedding['embedding']],
        n_results=3
    )

    print(f"\n[INFO] Test query: '{test_query}'")
    print(f"[INFO] Retrieved {len(results['documents'][0])} results:")
    for i, doc in enumerate(results['documents'][0], 1):
        print(f"  {i}. {doc[:100]}...")

    print("\n" + "="*60)
    print("[SUCCESS] Chroma vector store setup complete!")
    print("="*60)
    print(f"\nVector store details:")
    print(f"  - Location: {CHROMA_DB_DIR}")
    print(f"  - Collection: {collection_name}")
    print(f"  - Documents: {count}")
    print(f"  - Embedding model: {ModelConfig.EMBEDDING_MODEL}")
    print(f"\nNext steps:")
    print(f"  - Implement agents using this vector store for RAG")

if __name__ == "__main__":
    setup_chroma_store()
