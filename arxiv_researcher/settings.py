from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel

class QueryDecomposerSettings(BaseModel):
    """QueryDecomposerエージェントの設定"""

    # タスク分解時の最小タスク
    min_decomposed_tasks: int = 3
    # タスク分解時の最大タスク
    max_decomposed_tasks: int = 5

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # 必須のAPIキー
    OPENAI_API_KEY: str
    COHERE_API_KEY: str
    JINA_API_KEY: str
    LANGSMITH_API_KEY: str

    # エージェントごとの設定インスタンス
    query_decomposer: QueryDecomposerSettings = QueryDecomposerSettings()

# グローバルスコープの設定インスタンス
settings = Settings()