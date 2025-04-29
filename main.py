from utils.utils import process_pdfs, create_vector_store
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def main():
    # Get directory configurations from environment variables
    files_directory = os.getenv('FILES_DIRECTORY', 'files')  # fallback to 'files' if not set
    persist_directory = os.getenv('VECTOR_STORE_DIRECTORY', 'vector_store')  # fallback to 'vector_store' if not set
    
    # Process all PDFs and get text chunks
    chunks = process_pdfs(files_directory)
    
    # Create and populate vector store
    vector_store = create_vector_store(
        documents=chunks,
        persist_directory=persist_directory
    )

if __name__ == '__main__':
    main()
