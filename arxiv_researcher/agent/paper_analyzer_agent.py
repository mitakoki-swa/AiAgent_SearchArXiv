import logging
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from arxiv_researcher.models import ReadingResult
from arxiv_researcher.chains.reading_chains import (
    CheckSufficiency,
    SetSection,
    Sufficiency,
    Summarizer
)
from arxiv_researcher.settings import settings

class PaperAnalyzerAgentInputState(TypedDict):
    goal: str
    reading_result: ReadingResult

class PaperAnalyzerAgentProcessingState(TypedDict):
    selected_section_indices: list[int]
    sufficiency: Sufficiency
    check_count: int

class PaperAnalyzerAgentOutputState(TypedDict):
    reading_result: ReadingResult

class PaperAnalyzerAgentState(
    PaperAnalyzerAgentInputState,
    PaperAnalyzerAgentOutputState,
    PaperAnalyzerAgentProcessingState
):
    pass


class PaperAnalyzerAgent:
    # 読み取りを行う最大セクション数
    MAX_SECTIONS = 5

    # 十分正をチェックする回数
    CHECK_COUNT = 3

    def __init__(self, llm: ChatOpenAI):
        self.set_section = SetSection(llm, max_sections=self.MAX_SECTIONS)
        self.check_sufficiency = CheckSufficiency(llm, check_count=self.CHECK_COUNT)
        self.summarizer = Summarizer(llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            PaperAnalyzerAgentState,
            input_schema=PaperAnalyzerAgentInputState,
            output_schema=PaperAnalyzerAgentOutputState
        )
        workflow.add_node("set_section", self.set_section)
        workflow.add_node("check_sufficiency", self.check_sufficiency)
        workflow.add_node("mark_as_not_related", self._mark_as_not_related)
        workflow.add_node("summarize", self.summarizer)

        workflow.set_entry_point("set_section")
        workflow.set_finish_point("summarize")
        workflow.set_finish_point("mark_as_not_related")

        return workflow.compile()

    def _mark_as_not_related(self, state: PaperAnalyzerAgentState) -> Command:
        reading_result = state.get("reading_result")

        # デバッグ用ログ
        logger = logging.getLogger(__name__)
        if reading_result:
            logger.info(f"======== title: {reading_result.paper.title} ========")
            logger.info(f"======== relevance_score: {reading_result.paper.relevance_score} ========")
            logger.info(f"======== is_related: {reading_result.is_related} ========")

        if reading_result is None:
            raise ValueError("reading_result is not set")
        reading_result.is_related = False
        return Command(
            update={"reading_result": reading_result}
        )

graph = PaperAnalyzerAgent(llm=settings.fast_llm).graph