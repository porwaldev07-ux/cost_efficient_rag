import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    
    OPENAI_API_KEY: str = "local-free-mode"
    
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    EMBEDDING_DIM: int = 384
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    QDRANT_PATH: str = "./local_qdrant_storage"

settings = Settings()