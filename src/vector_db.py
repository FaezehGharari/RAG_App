from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

EMBED_DIM = 2048
class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=EMBED_DIM):
        self.client = QdrantClient(url= url, timeout=30)
        self.collection= collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )

    def upsert(self, ids, vectors, payloads):
        points= [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k:int=5):
        query_points = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        )

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