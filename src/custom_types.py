import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    sources: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested: int

class RAGSearchResults(pydantic.BaseModel):
    context: list[str]
    sources: list[str]

class RAGQueryResult(pydantic.BaseModel):
    answer: str
    num_context: int
    sources: list[str]