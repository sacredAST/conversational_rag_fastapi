import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DB_USERNAME: str = "alexclayton298"
    DB_PASSWORD: str = "Y78eRcSwfCrOaWMr"
    DB_URL: str = "testcluster.weptpwq.mongodb.net"
    DB_NAME: str = "vector_db"
    CLUSTER_NAME: str = "testcluster"
    COLLECTION_NAME: str = "vectors"
    ATLAS_URL: str = f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@{DB_URL}/?retryWrites=true&w=majority&appName={CLUSTER_NAME}"
    HUGGINGFACE_EMBEDDING: str = "sentence-transformers/all-MiniLM-L6-v2"
    INDEX_NAME: str = "embedding"
    LLM_MODEL: str = "lzlv_70b"
    API_KEY: str = "62401f1a-849c-4920-8ed5-5ee342dd4405"
    API_URL: str = "https://api.novita.ai/v3/openai"
settings = Settings()