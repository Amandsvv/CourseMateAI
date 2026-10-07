import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

class VectorStoreManager:
    """
    Manages embedding generation, vector storage, persistence,
    and similarity searches using ChromaDB.
    """
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "coursemate_docs"
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Uses Mistral's 1024-dim embedding model
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        self._initialize_store()

    def _initialize_store(self):
        """Initializes connection to the local Chroma persistent storage."""
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def index_documents(self, documents: List[Document], batch_size: int = 250) -> int:
        """
        Embeds and indexes documents in batches to handle API rate limits.
        Returns the total number of documents added.
        """
        if not documents:
            print("No documents provided for indexing.")
            return 0
        
        total_docs = len(documents)
        print(f"Indexing {total_docs} chunks into ChromaDB '{self.collection_name}'...")
        
        # Batch insert to prevent timeout / rate limit spikes
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            self.vector_store.add_documents(documents=batch)
            print(f"  Indexed batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} items)")
        
        print(f"Indexing complete. Database persisted at '{self.persist_directory}'.")
        return total_docs

    def similarity_search_with_score(self, query: str, k: int = 4):
        """
        Performs similarity search and returns chunks with distance scores.
        Note: Lower distance = higher semantic similarity (L2 / Cosine distance).
        """
        return self.vector_store.similarity_search_with_score(query, k=k)

    def get_retriever(self, search_type: str = "similarity", k: int = 4):
        """
        Exposes a standard LangChain retriever interface for LCEL chains.
        search_type can be 'similarity' or 'mmr' (Maximal Marginal Relevance).
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )


if __name__ == "__main__":
    # Smoke-test Vector Store indexing and query retrieval
    manager = VectorStoreManager(persist_directory="./chroma_db_test")
    
    # Test data
    test_chunks = [
        Document(
            page_content="Gradient descent optimizes loss functions by iteratively moving in the direction of steepest descent.",
            metadata={"file_name": "optimization.pdf", "page_number": 12}
        ),
        Document(
            page_content="Backpropagation computes the gradient of the loss function with respect to the weights of the network.",
            metadata={"file_name": "neural_networks.pdf", "page_number": 45}
        )
    ]
    
    manager.index_documents(test_chunks)
    
    query = "How do neural networks calculate weight gradients?"
    results = manager.similarity_search_with_score(query, k=1)
    
    print("\n--- Search Test ---")
    print(f"Query: {query}")
    for doc, score in results:
        print(f"Distance Score: {score:.4f}")
        print(f"Source: {doc.metadata['file_name']}, Page {doc.metadata['page_number']}")
        print(f"Content: {doc.page_content}")