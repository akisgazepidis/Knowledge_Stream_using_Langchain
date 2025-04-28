import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

def count_vectors(persist_directory: str = "vector_store"):
    """
    Connect to ChromaDB and count the number of vectors stored.
    """
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Get the collection
        collection = client.get_collection("pdf_collection")
        
        # Get collection count
        count = collection.count()
        print(f"\nNumber of vectors stored in the database: {count}")
        
    except Exception as e:
        print(f"Error accessing the vector store: {str(e)}")

def get_first_result(persist_directory: str = "vector_store"):
    """
    Retrieve the first result from the vector database.
    """
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Get the collection
        collection = client.get_collection("pdf_collection")
        
        # Get first result (using a generic query to get any document)
        results = collection.query(
            query_texts=[""],
            n_results=1
        )
        
        if results['documents'] and results['documents'][0]:
            print("\nFirst document in the database:")
            print("-" * 50)
            print(results['documents'][0][0])
            print("\nMetadata:")
            print(f"Distance: {results['distances'][0][0]}")
            print(f"ID: {results['ids'][0][0]}")
        else:
            print("No documents found in the database.")
        
    except Exception as e:
        print(f"Error accessing the vector store: {str(e)}")

def get_first_vector(persist_directory: str = "vector_store"):
    """
    Retrieve the vector embedding of the first document in the database.
    """
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Get the collection
        collection = client.get_collection("pdf_collection")
        
        # Get first result with embeddings included
        results = collection.get(
            limit=1,
            include=['embeddings']
        )
        
        if len(results['embeddings']) > 0:
            print("\nVector embedding of the first document:")
            print("-" * 50)
            embedding = np.array(results['embeddings'][0])
            print(f"Shape: {embedding.shape}")
            print(f"First 5 dimensions: {embedding[:5]}")
            print(f"Vector norm: {np.linalg.norm(embedding):.4f}")
            return embedding
        else:
            print("No vectors found in the database.")
            return None
        
    except Exception as e:
        print(f"Error accessing the vector store: {str(e)}")
        return None

if __name__ == "__main__":
    count_vectors()
    get_first_vector()