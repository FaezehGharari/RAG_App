import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
from data_loader import load_and_chunk_pdf, embed_text
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGSearchResults, RAGUpsertResult
import os
import datetime
from config import settings

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn") ,
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    throttle=inngest.Throttle(
        limit=2, period=datetime.timedelta(minutes=1)
    ),
    rate_limit=inngest.RateLimit(
        limit=1,
        period=datetime.timedelta(hours=4),
        key="event.data.source_id",
  ),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, sources=source_id)

    def _upsert(chunks_and_sources: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_sources.chunks
        sources = chunks_and_sources.sources
        vectors = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{sources} : {i}")) for i in range(len(chunks))]
        payloads = [{"sources" : sources, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vectors, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_sources = await ctx.step.run("chunks_and_resources", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed_and_upsert", lambda: _upsert(chunks_and_sources), output_type=RAGUpsertResult)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: search pdf",
    trigger=inngest.TriggerEvent(event= "search_pdf")
)
async def rag_search_pdf(ctx : inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResults:
        query_vector = embed_text([question])[0]
        storage = QdrantStorage()
        found = storage.search(query_vector,top_k)
        return RAGSearchResults(context= found["context"], sources= found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed_query_and_search", lambda: _search(question , top_k), output_type=RAGSearchResults)

    context_block = "\n\n".join(f"- {c}" for c in found.context)
    user_content = (
        "use the following context to answer the question\n\n"
        f"context:\n{context_block}\n\n"
        f"question:\n{question}\n\n"
        "answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model= settings.llm_model
    )

    response = await ctx.step.ai.infer(
        "llm_answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages":[
                {"role": "system", "content": "you answer questions using only the provided context"},
                {"role": "user", "content": user_content}
            ]
        }
    )

    answer= response["choices"][0]["message"]["content"].strip()
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.context)}


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_search_pdf])