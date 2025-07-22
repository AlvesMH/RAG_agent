import os
import uuid
import json
import requests
from typing import List, Tuple

import gradio as gr
from fpdf import FPDF
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # pip install -U langchain-chroma

# Load environment variables
load_dotenv()
SEA_LION_API_KEY = os.getenv("SEA_LION_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# Configuration
SEA_LION_BASE_URL = "https://api.sea-lion.ai/v1/chat/completions"
DEFAULT_SEALION_MODEL = "aisingapore/Gemma-SEA-LION-v3-9B-IT"

HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBED_API = (
    f"https://router.huggingface.co/"
    f"hf-inference/models/{HF_EMBED_MODEL}/pipeline/feature-extraction"
)

CHROMA_DB_DIR = "chroma_db"

# SeaLion helper functions
def _call_sealion(messages: List[dict], model_name: str, temperature: float = 0.7) -> str:
    if not SEA_LION_API_KEY:
        return "Error: SEA_LION_API_KEY not set"
    headers = {
        "Authorization": f"Bearer {SEA_LION_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"model": model_name, "messages": messages, "temperature": temperature}
    try:
        resp = requests.post(SEA_LION_BASE_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [])[0].get("message", {}).get("content", "").strip()
    except Exception as e:
        return f"SeaLion API error: {e}"


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}

# Custom embedder using Hugging Face Inference API
class HFEmbedder:
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        payload = {"inputs": texts}
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        resp.raise_for_status()
        # The API returns a list of embeddings (one per input)
        return resp.json()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# Initialize vector store
hf_embedder = HFEmbedder(api_url=HF_EMBED_API, token=HUGGINGFACE_API_TOKEN)
vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=hf_embedder)


# Text splitter for PDF content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def ingest_pdfs(files: List[str]) -> str:
    if not files:
        return "No files uploaded."
    from PyPDF2 import PdfReader
    count = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            chunks = text_splitter.split_text(text)
            docs = [
                Document(
                    page_content=chunk,
                    metadata={"source": filename, "title": filename, "page": page_num}
                )
                for chunk in chunks if chunk.strip()
            ]
            if docs:
                vectordb.add_documents(docs)
        count += 1
    return f"Ingested {count} document(s) into ChromaDB."


def retrieve_excerpts(query: str, k: int = 3) -> str:
    if not query:
        return ""
    docs = vectordb.similarity_search(query, k=k)
    if not docs:
        return "No relevant excerpts found."
    excerpts = []
    for doc in docs:
        src = doc.metadata.get("source", "")
        text = doc.page_content.replace("\n", " ")
        excerpts.append(f"**{src}**: {text}")
    return "\n\n".join(excerpts)


def generate_response(query: str, k: int = 3) -> Tuple[str, str]:
    if not query:
        return "", ""
    docs = vectordb.similarity_search(query, k=k)
    context = "\n\n".join(
        f"Source: {d.metadata.get('title', '')} (page {d.metadata.get('page', '?')})\n" + d.page_content.replace("\n", " ")
        for d in docs
    )
    system_msg = _msg("system", "You are a helpful assistant.")
    user_msg = _msg("user", f"Use the following context to answer the question.\n\n{context}\n\nQuestion: {query}")
    answer = _call_sealion([system_msg, user_msg], model_name=DEFAULT_SEALION_MODEL)
    sources_info = "\n\nRelevant sources:\n" + "\n".join(
        f"- {d.metadata.get('title', '')}, page {d.metadata.get('page', '?')}" for d in docs
    )
    full_answer = answer + sources_info
    pdf_filename = f"response_{uuid.uuid4().hex}.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in full_answer.split("\n"):
        pdf.multi_cell(0, 10, line)
    pdf.output(pdf_filename)
    return full_answer, pdf_filename


def list_documents() -> str:
    results = vectordb._collection.get(include=["metadatas"])  # type: ignore
    metadatas = results.get("metadatas", [])
    titles = sorted({md.get("title", "") for md in metadatas if md.get("title", "")})
    if not titles:
        return "No documents in database."
    return "\n".join(titles)


def main():
    port = int(os.environ.get("PORT", 7860))
    with gr.Blocks() as demo:
        gr.Markdown("# Simple RAG Agent (SeaLion + HF)")

        with gr.Row():
            upload = gr.File(label="Upload PDF Documents", file_count="multiple", type="filepath")
            ingest_btn = gr.Button("Add to VectorDB")
            list_btn = gr.Button("List Documents")
        ingest_output = gr.Textbox(label="Ingest Status", lines=1)
        list_output = gr.Textbox(label="Document Titles", lines=5)
        ingest_btn.click(fn=ingest_pdfs, inputs=upload, outputs=ingest_output)
        list_btn.click(fn=list_documents, inputs=None, outputs=list_output)

        with gr.Row():
            query = gr.Textbox(label="Enter your question", lines=2)
            retrieve_btn = gr.Button("Retrieve Excerpts")
        retrieve_output = gr.Textbox(label="Retrieved Excerpts", lines=10)
        retrieve_btn.click(fn=retrieve_excerpts, inputs=query, outputs=retrieve_output)

        with gr.Row():
            gen_btn = gr.Button("Generate Response")
            download_file = gr.File(label="Download Response PDF", file_count="single", type="filepath")
        answer_output = gr.Textbox(label="Port Agent Response", lines=10)
        gen_btn.click(fn=generate_response, inputs=query, outputs=[answer_output, download_file])
        pass

        demo.launch(server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    main()
