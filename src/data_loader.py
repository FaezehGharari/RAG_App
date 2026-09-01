from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from config import settings
from pathlib import Path
import logging

load_dotenv()

logger = logging.getLogger("uvicorn")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=30
)

splitter = SentenceSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

def load_and_chunk_pdf(path: str):
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError("pdf file not found ", pdf_path)
    
    try:
        docs = PDFReader().load_data(file=path)
    except Exception:
        logger.exception("Failed to load PDF %s", pdf_path)
        raise

    text = [d.text for d in docs if getattr(d,"text", None)]
    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_text(texts: list[str]) -> list[list[float]]:
    try:
        responce = client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
            encoding_format="float"
        )
        return [item.embedding for item in responce.data]
    except Exception:
        logger.exception("Embedding generation failed. texts_count=%d",len(texts),)
        raise