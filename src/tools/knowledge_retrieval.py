"""
Knowledge retrieval tool using Chroma vector store for RAG.
"""
import json
from typing import List, Dict
from chromadb import Client
from chromadb.config import Settings

from config import CHROMA_DB_DIR, AgentConfig, ModelConfig, KNOWLEDGE_BASE_FILE
from utils.llm_client import get_embedding_function

class KnowledgeRetriever:
    """Retrieval-Augmented Generation using Chroma vector store"""

    def __init__(self):
        """Initialize Chroma client and collection"""
        self.client = Client(Settings(
            persist_directory=str(CHROMA_DB_DIR),
            anonymized_telemetry=False
        ))

        # Try to get existing collection or use fallback
        try:
            self.collection = self.client.get_collection(
                name=AgentConfig.CHROMA_COLLECTION_NAME
            )
            print(f"[OK] Loaded existing collection with {self.collection.count()} documents")
        except:
            print(f"[INFO] Collection not found, using keyword-based fallback")
            self.collection = None

        # Load knowledge base for fallback
        with open(KNOWLEDGE_BASE_FILE, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)

        print(f"[OK] Knowledge retriever initialized with {len(self.knowledge_base)} KB entries")

        # Initialize embedding function (lazy loaded when needed)
        self._embed_fn = None

    def _get_embed_function(self):
        """Lazy load embedding function"""
        if self._embed_fn is None:
            try:
                self._embed_fn = get_embedding_function()
            except Exception as e:
                print(f"[WARN] Could not initialize embeddings: {e}")
                self._embed_fn = None
        return self._embed_fn

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents from vector store.

        Args:
            query: User query
            top_k: Number of documents to retrieve (default from config)

        Returns:
            List of relevant documents with metadata
        """
        if top_k is None:
            top_k = AgentConfig.RETRIEVAL_TOP_K

        # If collection not available, use keyword search fallback
        if self.collection is None:
            return self._fallback_search(query, top_k)

        try:
            # Get embedding function
            embed_fn = self._get_embed_function()
            if embed_fn is None:
                return self._fallback_search(query, top_k)

            # Generate query embedding using unified client
            query_embedding = embed_fn(query, task_type="retrieval_query")

            # Query vector store
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )

            # Format results
            documents = []
            for i in range(len(results['documents'][0])):
                doc = {
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                documents.append(doc)

            return documents
        except Exception as e:
            print(f"[WARN] Vector search failed: {e}, using fallback")
            return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        matches = []

        for kb_entry in self.knowledge_base:
            # Simple keyword matching
            question_lower = kb_entry['question'].lower()
            answer_lower = kb_entry['answer'].lower()

            score = 0
            for word in query_lower.split():
                if len(word) > 3:  # Skip short words
                    if word in question_lower:
                        score += 2
                    if word in answer_lower:
                        score += 1

            if score > 0:
                matches.append({
                    "content": f"Q: {kb_entry['question']}\nA: {kb_entry['answer']}",
                    "metadata": kb_entry,
                    "score": score
                })

        # Sort by score and return top_k
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k] if matches else [{"content": "No relevant information found", "metadata": {}, "score": 0}]

    def search_by_intent(self, intent: str) -> List[Dict]:
        """Search knowledge base by intent category"""
        matching_docs = [
            kb for kb in self.knowledge_base
            if kb['intent'].lower() == intent.lower() or kb['category'].lower() == intent.lower()
        ]
        return matching_docs[:AgentConfig.RETRIEVAL_TOP_K]

    def get_formatted_context(self, query: str, top_k: int = None) -> str:
        """
        Retrieve and format context for LLM prompt.

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Formatted context string
        """
        documents = self.retrieve(query, top_k)

        if not documents or (len(documents) == 1 and documents[0]['content'] == "No relevant information found"):
            return "No relevant information found in knowledge base."

        context = "Relevant information from knowledge base:\n\n"
        for i, doc in enumerate(documents, 1):
            context += f"{i}. {doc['content']}\n\n"

        return context.strip()

# Global instance
knowledge_retriever = KnowledgeRetriever()
