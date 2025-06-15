import json
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseIndexer:
    def __init__(self, collection_name: str = "kb_index"):
        """Initialize the KB indexer with ChromaDB and sentence transformer."""
        self.collection_name = collection_name
        
        # Initialize ChromaDB (using local persistent storage)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Initialize sentence transformer for embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Retrieved existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def load_kb_data(self, file_path: str) -> List[Dict]:
        """Load the knowledge base data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} entries from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise
    
    def compute_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings for the given texts."""
        embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def index_knowledge_base(self, kb_data: List[Dict]) -> None:
        """Index the knowledge base entries into the vector store."""
        # Extract texts for embedding
        texts = [entry['answer_snippet'] for entry in kb_data]
        
        # Compute embeddings
        logger.info("Computing embeddings...")
        embeddings = self.compute_embeddings(texts)
        
        # Prepare data for ChromaDB
        ids = [entry['doc_id'] for entry in kb_data]
        documents = texts
        metadatas = [
            {
                'source': entry['source'],
                'last_updated': entry['last_updated'],
                'question': entry['question']
            }
            for entry in kb_data
        ]
        
        # Upsert to collection
        logger.info("Upserting to vector store...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Successfully indexed {len(kb_data)} entries")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the knowledge base for relevant entries."""
        # Compute query embedding
        query_embedding = self.compute_embeddings([query])[0]
        
        # Search the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        hits = []
        for i in range(len(results['ids'][0])):
            hits.append({
                'doc_id': results['ids'][0][i],
                'answer_snippet': results['documents'][0][i],
                'source': results['metadatas'][0][i]['source'],
                'distance': results['distances'][0][i]
            })
        
        return hits
    
    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_documents': count
        }

def main():
    """Main function to index the knowledge base."""
    # Initialize indexer
    indexer = KnowledgeBaseIndexer()
    
    # Load KB data
    kb_data = indexer.load_kb_data('self_critique_loop_dataset.json')
    
    # Index the data
    indexer.index_knowledge_base(kb_data)
    
    # Print collection info
    info = indexer.get_collection_info()
    print(f"Collection Info: {info}")
    
    # Test search
    print("\n--- Testing Search ---")
    test_query = "What are best practices for caching?"
    results = indexer.search(test_query, top_k=3)
    
    print(f"Query: {test_query}")
    for i, hit in enumerate(results, 1):
        print(f"{i}. [{hit['doc_id']}] {hit['answer_snippet'][:100]}...")
        print(f"   Source: {hit['source']}, Distance: {hit['distance']:.4f}")

if __name__ == "__main__":
    main()