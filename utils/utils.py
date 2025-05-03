from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain_huggingface import HuggingFaceEndpoint
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
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
    """
    # Initialize the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(path=persist_directory)
    
    try:
        # Try to get existing collection first
        collection = client.get_collection(name="pdf_collection")
        print("Found existing collection")
    except:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name="pdf_collection",
            metadata={"hnsw:space": "cosine"}
        )
        print("Created new collection")
    
    # Extract text content from Document objects
    texts = [doc.page_content for doc in documents]
    
    # Add documents to the collection
    if texts:
        collection.add(
            documents=texts,
            embeddings=model.encode(texts).tolist(),
            ids=[f"doc_{i}" for i in range(len(texts))]
        )
        print(f"Added {len(texts)} documents to the collection")
    
    return client

def setup_agent(files_directory: str = "files", persist_directory: str = "vector_store"):
    """
    Set up a LangChain agent with PDF processing and vector storage tools using Hugging Face model.
    
    Args:
        files_directory (str): Directory containing PDF files
        persist_directory (str): Directory for vector store persistence
    """
    # Wrap tools in LangChain Tool objects
    tools = [
        Tool(
            name="ProcessPDFs",
            func=lambda _: process_pdfs(files_directory),
            description=f"Process PDF documents from the {files_directory} directory and split them into chunks. This is the first step in the pipeline."
        ),
        Tool(
            name="StoreVectors",
            func=lambda chunks: create_vector_store(chunks, persist_directory),
            description="Store the provided chunks as vector embeddings in ChromaDB. Use this after ProcessPDFs to complete the pipeline."
        )
    ]

    # Initialize LLM using Hugging Face's Endpoint
    llm = HuggingFaceEndpoint(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        temperature=0.1,
        max_new_tokens=512,
        do_sample=False,  # Moved from model_kwargs to direct parameter
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    # Create memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output"
    )

    # Initialize agent with specific configuration
    return initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=2  # Limit iterations to prevent unnecessary operations
    )