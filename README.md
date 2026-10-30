# Knowledge Stream using LangChain

An intelligent, real-time knowledge ingestion and retrieval pipeline built with **LangChain**, designed to process, chunk, embed, and query streaming data sources efficiently.

## 🚀 Key Features

* **Advanced Text Splitting & Chunking:** Implements semantic and recursive chunking strategies to optimize document context retention for downstream retrieval.
* **Vector Store Integration:** Seamlessly connects with high-performance vector databases for fast similarity search and embedding storage.
* **Retrieval-Augmented Generation (RAG):** Integrates prompt engineering and LLM orchestration to answer queries backed by ingested knowledge bases.
* **Modular Pipeline Architecture:** Cleanly decouples document loaders, embedding generators, vector storage, and generation chains.

## 🛠️ Tech Stack

* **Orchestration:** LangChain
* **Language:** Python
* **Embeddings & LLMs:** OpenAI / Google Gemini (configurable)
* **Vector Storage:** Chroma / FAISS

## 📁 Project Structure

```text
Knowledge_Stream_using_Langchain/
├── data/           # Raw and processed documents
├── src/            # Core pipeline logic (ingestion, chunking, retrieval)
├── notebooks/      # Exploratory data analysis and prototyping
├── main.py         # Pipeline entry point
└── requirements.txt # Project dependencies