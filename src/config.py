from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    cohere_api_key: str
    database_url: str 
    embedding_model: int = 384
    llm_model: str = "models/gemini-3.5-flash-lite"
    retrieval_top_k: int = 20
    rerank_top_n: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 50
    google_api_key: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_base_url: str
    groq_api_key: str

    class Config:
        env_file = ".env"

settings = Settings()