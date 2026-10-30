# Knowledge Stream using LangChain

An intelligent, real-time knowledge ingestion and retrieval pipeline built with LangChain, designed to process, chunk, embed, and query streaming data sources efficiently.

## Key Features

* Advanced Text Splitting & Chunking: Implements semantic and recursive chunking strategies to optimize document context retention for downstream retrieval.
* Vector Store Integration: Seamlessly connects with high-performance vector databases for fast similarity search and embedding storage.
* Retrieval-Augmented Generation (RAG): Integrates prompt engineering and LLM orchestration to answer queries backed by ingested knowledge bases.
* Modular Pipeline Architecture: Cleanly decouples document loaders, embedding generators, vector storage, and generation chains.

## Tech Stack

* Orchestration: LangChain
* Language: Python
* Embeddings & LLMs: OpenAI / Google Gemini (configurable)
* Vector Storage: Chroma / FAISS

## Project Structure


```text
Knowledge_Stream_using_Langchain/
├── data/           # Raw and processed documents
├── src/            # Core pipeline logic (ingestion, chunking, retrieval)
├── notebooks/      # Exploratory data analysis and prototyping
├── main.py         # Pipeline entry point
└── requirements.txt # Project dependencies
```

## Getting Started

### Prerequisites

* Python 3.10+
* Virtual environment (recommended)

### Installation

1. Clone the repository:
   git clone https://github.com/akisgazepidis/Knowledge_Stream_using_Langchain.git
   cd Knowledge_Stream_using_Langchain

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
   pip install -r requirements.txt

4. Set up your environment variables (create a .env file):
   OPENAI_API_KEY=your_api_key_here

## Usage

Run the main pipeline script to initialize ingestion and query the knowledge base:
python main.py

## License

Distributed under the MIT License. See LICENSE for more information.