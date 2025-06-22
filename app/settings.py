from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
import os
from dotenv import load_dotenv

_package_root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_dotenv_path=os.path.join(_package_root_directory, ".env")
load_dotenv(dotenv_path=_dotenv_path, override=True)
class TextSplitterConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int

class OllamaConfig(BaseModel):
    host: str
    embedding_model: str

class LangfuseConfig(BaseModel):
    public_key: str
    secret_key: str
    host: str

class LLMConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter="__")
    package_root_directory: str = _package_root_directory
    text_splitter: TextSplitterConfig = TextSplitterConfig(chunk_size=500, chunk_overlap=50)
    pdf_path: str = os.path.join(_package_root_directory, "pdfs", "vscodeInstallation.pdf")
    ollama: OllamaConfig
    enable_metrics: bool = False
    llm: LLMConfig = LLMConfig()

    # required environment configurations
    openai_api_key: str
    langfuse: LangfuseConfig



SETTINGS = Settings() # type: ignore

