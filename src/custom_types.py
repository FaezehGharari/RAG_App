import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    sources: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResults(pydantic.BaseModel):
    context: list[str]
    sources: list[str]
