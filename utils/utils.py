from typing import List, Dict
from langchain_community.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
import chromadb
import os

def load_pdf(file_path: str) -> List[str]:
    """
    Load and extract text from a PDF file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found at {file_path}")
    
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def split_text(text_pages: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(text_pages)
    return chunks

def process_pdfs(directory: str) -> List[str]:
    """
    Process all PDF files in the given directory.
    """
    all_chunks = []
    
    # Get all PDF files from the directory
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the directory!")
        return all_chunks
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(directory, pdf_file)
        print(f"Processing {pdf_file}...")
        
        # Load and process each PDF
        pages = load_pdf(pdf_path)
        chunks = split_text(pages)
        all_chunks.extend(chunks)
    
    return all_chunks

def create_vector_store(documents: List[str], persist_directory: str = "vector_store"):
    """
    Create and populate a ChromaDB vector store with the provided documents.
    
    Args:
        documents (List[str]): List of text chunks to store
        persist_directory (str): Directory to persist the vector store
    
    Returns:
        chromadb.Client: Configured ChromaDB client
    """
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client
    client = chromadb.Client(Settings(
        persist_directory=persist_directory,
        anonymized_telemetry=False
    ))
    
    # Create or get collection
    collection = client.create_collection(
        name="pdf_collection",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Add documents to the collection
    if documents:
        collection.add(
            documents=documents,
            embeddings=model.encode(documents).tolist(),
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
    
    return client