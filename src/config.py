from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    embedding_model: str = "nvidia/nemotron-3-embed-1b:free"
    embedding_dimension: int = 2048

    llm_model: str = "google/gemma-4-31b-it:free"

    chunk_size: int = 1000
    chunk_overlap: int = 200

settings = Settings()
