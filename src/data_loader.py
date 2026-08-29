from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    timeout=30
)

EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
EMBED_DIM = 2048

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    text = [d.text for d in docs if getattr(d,"text", None)]
    chunks = []
    for t in text:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_text(texts: list[str]) -> list[list[float]]:
    responce = client.embeddings.create(
        input=texts,
        model=EMBED_MODEL,
        encoding_format="float"
    )
    return [item.embedding for item in responce.data]