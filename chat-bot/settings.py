from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    BOT_TOKEN: str
    LANGFLOW_HOST: str
    LANGFLOW_PORT: str
    OLLAMA_HOST: str
    OLLAMA_PORT: str
    OLLAMA_LLM: str
    OLLAMA_EMBEDDING_MODEL: str
    CHROMA_DIR: str
    SEPARATORS: list[str]
    CONTEXT_FILE_PATH: str
    CHUNK_SIZE: int

    @property
    def lagflow_host(self):
        return f"http://{self.LANGFLOW_HOST}:{self.LANGFLOW_PORT}"

    @property
    def ollama_host(self):
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

    class Config:
        env_file = ".env"


settings = Settings()