from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import cohere


class ArxivResearcherSettings(BaseModel):
    """ArxivResearcherエージェントの設定"""

    # 追加情報取得のための最大リトライ回数
    max_evaluation_retry_count: int = 3

class QueryDecomposerSettings(BaseModel):
    """QueryDecomposerエージェントの設定"""

    # タスク分解時の最小タスク
    min_decomposed_tasks: int = 3
    # タスク分解時の最大タスク
    max_decomposed_tasks: int = 5


class ArxivSearchAgentSettings(BaseModel):
    """ArxivSearchAgentエージェントの設定"""

    # 検索失敗時の最大リトライ数
    max_retries: int = 3
    # 1回の検索で取得する最大論文数
    max_search_results: int = 10
    # 詳細分析する最大論文数
    max_papers: int = 3
    # PDFからmarkdown変換時の並列処理数
    max_workers: int = 3


class LangGraphSettings(BaseModel):
    """LangGraph関連の設定"""

    # ノードの最大実行回数制限
    max_recursion_limit: int = 1000


class ModelSettings(BaseModel):
    """モデル関連の設定"""

    # 高速・軽量なタスク用のOpenAIモデル
    openai_fast_model: str = "gpt-4o-mini"
    # 複雑なタスク用のOpenAIモデル
    openai_smart_model: str = "gpt-4o"
    # レポート生成用のAnthropicモデル
    anthropic_model: str = "claude-3-7-sonnet-20250219"
    # リランキング用のCohereモデル
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    # 生成時の温度パラメータ
    temperature: float = 0.0


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
    ANTHROPIC_API_KEY: str

    # デバッグモード設定
    debug: bool = True

    # エージェントごとの設定インスタンス
    arxiv_researcher: ArxivResearcherSettings = ArxivResearcherSettings()
    query_decomposer: QueryDecomposerSettings = QueryDecomposerSettings()
    arxiv_search_agent: ArxivSearchAgentSettings = ArxivSearchAgentSettings()
    model: ModelSettings = ModelSettings()
    langgraph: LangGraphSettings = LangGraphSettings()



    @property
    def llm(self) -> ChatOpenAI:
        """複雑なタスク用のLLMインスタンスを返す"""
        return ChatOpenAI(
            model=self.model.openai_smart_model,
            temperature=self.model.temperature,
        )


    @property
    def fast_llm(self) -> ChatOpenAI:
        """高速・軽量なタスク用のLLMインスタンスを返す"""
        return ChatOpenAI(
            model=self.model.openai_fast_model,
            temperature=self.model.temperature
        )


    @property
    def reporter_llm(self) -> ChatAnthropic:
        """レポート生成用のLLMインスタンスを返す"""
        return ChatAnthropic(
            model=self.model.anthropic_model,
            temperature=self.model.temperature,
            max_tokens=8_192,
        )


    @property
    def cohere_client(self) -> cohere.Client:
        """Cohereクライアントインスタンスを返す"""
        return cohere.Client(api_key=self.COHERE_API_KEY)

# グローバルスコープの設定インスタンス
settings = Settings()