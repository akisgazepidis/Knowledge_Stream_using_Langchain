# PDF to Vector Database

A simple pipeline for converting PDF documents into searchable vector embeddings using LangChain and ChromaDB.

## Workflow
1. PDF files are loaded from the `files/` directory
2. Text is extracted and split into chunks
3. Chunks are converted to vectors using sentence-transformers
4. Vectors are stored in a ChromaDB database

## Project Structure
```
.
├── files/          # Place your PDFs here
├── scripts/        # Utility scripts for querying
├── utils/          # Core processing functions
├── vector_store/   # ChromaDB storage (generated)
├── main.py         # Main processing pipeline
└── requirements.txt
```

## Components
- `main.py`: Executes the PDF processing pipeline
- `utils/utils.py`: Contains core functions for PDF processing and vector storage
- `scripts/query_embedding.py`: Tools for querying the vector database
- `requirements.txt`: Project dependencies
