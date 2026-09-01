from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from config import settings
import logging

logger = logging.getLogger("uvicorn")

class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=settings.embedding_dimension):
        self.client = QdrantClient(url= url, timeout=30)
        self.collection= collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        try:
            points= [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
            self.client.upsert(self.collection, points=points)
        except Exception:
            logger.exception("Qdrant upsert failed!")
            raise

    def search(self, query_vector, top_k:int=5):
        try:
            query_points = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                with_payload=True,
                limit=top_k
            )
        except Exception:
            logging.exception("Failed to fetch query points!")
            raise

        results= query_points.points
        context = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None)
            text = payload.get("text", "")
            source = payload.get("sources", "")
            if text:
                context.append(text)
                sources.add(source)

        return {"context": context, "sources": list(sources)}