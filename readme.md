## Simple Retrieval-Augmented Generation (RAG) Agent 

### LIVE-DEMO: 
### https://simple-rag-agent.onrender.com

A simple Retrieval-Augmented Generation (RAG) agent that:

* **Ingests** PDF documents into a Chroma vector store
* **Embeds** text via Hugging Face Inference API (`all-MiniLM-L6-v2`)
* **Retrieves** relevant excerpts on user queries
* **Generates** contextualized answers using SeaLion LLM (e.g. `Gemma-SEA-LION-v3-9B-IT`)
* **Exports** responses as downloadable PDF files

*Built with Python, Gradio, LangChain, ChromaDB, PyPDF2 & FPDF*

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [API Reference](#api-reference)
8. [Contributing](#contributing)
9. [License](#license)

---

## Features

* **PDF Ingestion**: Splits each page into 1,000-character chunks (200-character overlap) for efficient indexing.
* **Embeddings**: Uses the Hugging Face Inference API endpoint for `sentence-transformers/all-MiniLM-L6-v2` to generate vector embeddings.
* **Similarity Search**: Retrieves top-k most similar document chunks via ChromaDB.
* **Contextual Q\&A**: Builds a prompt including retrieved context and queries SeaLion’s hosted LLM API for answers.
* **Downloadable Reports**: Packages answers (with listed source titles/pages) into a PDF for offline reference.

---

## Prerequisites

* Python 3.8+
* Docker (optional, for running a local ChromaDB instance)
* A SeaLion API key (`SEA_LION_API_KEY`)
* A Hugging Face Inference API token (`HUGGINGFACE_API_TOKEN`)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/AlvesMH/RAG_agent.git
   cd simple-rag-agent
   ```

2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\\Scripts\\activate.bat  # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Run ChromaDB in Docker**

   ```bash
   docker run -d --name chroma -p 8000:8000 -v $(pwd)/chroma_db:/chroma chromadb/chromadb:latest
   ```

---

## Configuration

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
# Then open .env and add:
# SEA_LION_API_KEY=your_sealion_key
# HUGGINGFACE_API_TOKEN=your_hf_token
```

* **`SEA_LION_API_KEY`**: Used to authenticate calls to SeaLion’s chat/completions endpoint.
* **`HUGGINGFACE_API_TOKEN`**: Grants access to the Hugging Face Inference API for embeddings.

---

## Usage

1. **Launch the Gradio app**

   ```bash
   python Simple_RAG_Agent.py
   ```

   This will open a local web interface (usually at `http://localhost:7860`).

2. **Ingest PDFs**

   * Click **Upload PDF Documents** → select one or more `.pdf` files.
   * Click **Add to VectorDB** → see ingestion status.
   * Check existing docs via **List Documents**.

3. **Retrieve Excerpts**

   * Enter a question in the **Enter your question** box.
   * Click **Retrieve Excerpts** to view top-k matching chunks.

4. **Generate Full Response**

   * With your query entered, click **Generate Response**.
   * The agent will call SeaLion, stitch context into an answer, display it, and save a PDF you can download.

---

## Project Structure

```
├── .env.example            # Example env file
├── requirements.txt        # Python dependencies
├── Simple_RAG_Agent.py     # Main application script
├── chroma_db/              # Vectorstore
├── README.md               # This file
               
```

---

## API Reference

### SeaLion Chat Completion

* **Endpoint**: `POST https://api.sea-lion.ai/v1/chat/completions`
* **Payload**: `{ model, messages, temperature }`
* **Response**: `{ choices: [ { message: { content } } ] }`

### Hugging Face Feature Extraction

* **Endpoint**: `POST https://api-inference.huggingface.co/pipeline/feature-extraction/{model}`
* **Payload**: `{ inputs: [...texts] }`
* **Response**: `[[float...], ...]` embeddings

---

## License

This project is distributed under the MIT License. See [LICENSE](LICENSE) for more details.

Built with ❤️ in Singapore