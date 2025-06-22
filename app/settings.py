from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, SecretStr
import os
from dotenv import load_dotenv
from enum import Enum

_package_root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dotenv_path = os.path.join(_package_root_directory, ".env")
load_dotenv(dotenv_path=_dotenv_path, override=True)


class TextSplitterConfig(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50


class OllamaConfig(BaseModel):
    host: str = "http://localhost:11535"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3:8b"
    temperature: float = 0.5

class OpenaiConfig(BaseModel):
    api_key: SecretStr = SecretStr("SOME-API-KEY")
    embedding_model: str = "text-embedding-ada-002"
    chat_model: str = "gpt-4"
    temperature: float = 0.5

class LangfuseConfig(BaseModel):
    public_key: str
    secret_key: str
    host: str

class Provider(str, Enum):
    ollama = "ollama"
    openai = "openai"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    package_root_directory: str = _package_root_directory
    pdf_path: str = os.path.join(_package_root_directory, "pdfs", "thinkpython2.pdf")
    enable_metrics: bool = False
    provider: Provider = Provider.ollama

    text_splitter: TextSplitterConfig = TextSplitterConfig()
    ollama: OllamaConfig = OllamaConfig()
    openai: OpenaiConfig = OpenaiConfig() # must define at least the api_key in .env file
    langfuse: LangfuseConfig # must define all in .env file

SETTINGS = Settings()  # type: ignore
