from utils.utils import process_pdfs, create_vector_store

def main():
    # Directory configurations
    files_directory = "files"
    persist_directory = "vector_store"
    
    # Process all PDFs and get text chunks
    chunks = process_pdfs(files_directory)

    # print(chunks[0])  
    # print(len(chunks)) 
    
    # Create and populate vector store
    vector_store = create_vector_store(
        documents=chunks,
        persist_directory=persist_directory
    )

if __name__ == '__main__':
    main()
