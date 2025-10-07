import operator
import logging
from typing_extensions import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from arxiv_researcher.models import ReadingResult
from arxiv_researcher.searcher.arxiv_searcher import ArxivSearcher
from arxiv_researcher.chains.paper_processor_chain import PaperProcessor
from arxiv_researcher.agent.paper_analyzer_agent import PaperAnalyzerAgent
from arxiv_researcher.settings import settings

class PaperSearchAgentInpuState(TypedDict):
    goal: str
    tasks: list[str]

class PaperSearchAgentProcessState(TypedDict):
    processing_reading_results: Annotated[list[ReadingResult], operator.add]

class PaperSearchAgentOutputState(TypedDict):
    reading_results: list[ReadingResult]

class PaperSearchAgentState(
    PaperSearchAgentInpuState,
    PaperSearchAgentProcessState,
    PaperSearchAgentOutputState
):
    pass


class PaperSearchAgent:
    def __init__(self, llm: ChatOpenAI, searcher: ArxivSearcher):
        self.recursion_limit = settings.langgraph.max_recursion_limit
        self.max_workers = settings.arxiv_search_agent.max_workers
        self.llm = llm
        self.searcher = searcher
        self.paper_processor = PaperProcessor(searcher=self.searcher, max_workers=self.max_workers)
        self.paper_analyzer = PaperAnalyzerAgent(llm)
        self.graph = self._create_graph()

    def __call__(self) -> CompiledStateGraph:
        return self.graph

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            PaperSearchAgentState,
            input_schema=PaperSearchAgentInpuState,
            output_schema=PaperSearchAgentOutputState
        )
        workflow.add_node("search_papers", self.paper_processor)
        workflow.add_node("analyze_paper", self._analyze_paper)
        workflow.add_node("organize_results", self._organize_results)

        workflow.set_entry_point("search_papers")
        workflow.add_edge("analyze_paper", "organize_results")
        workflow.set_finish_point("organize_results")

        return workflow.compile()

    def _analyze_paper(self, state: dict) -> dict:
        output = self.paper_analyzer.graph.invoke(
            state,
            config={
                "recursion_limit": self.recursion_limit
            }
        )
        reading_result = output.get("reading_result")

        # デバッグ用ログ
        logger = logging.getLogger(__name__)
        if reading_result:
            logger.info("======== processing_reading_reusultあり _analyze_paper ========")
        else:
            logger.info("======== processing_reading_reusultなし _analyze_paper ========")

        return {
            "processing_reading_results": [reading_result] if reading_result else []
        }


    def _organize_results(self, state: dict) -> dict:
        processing_reading_results = state.get("processing_reading_results", [])
        reading_results = []

        # デバッグ用ログ
        logger = logging.getLogger(__name__)
        if processing_reading_results:
            logger.info("======== processing_reading_resultsあり organize_results ========")
            for processing_reading_result in processing_reading_results:
                print(f"is_related: {processing_reading_result.is_related}")
        else:
            logger.info("======== processing_reading_resultsなし organize_results ========")

        # 関連性のある論文のみをフィルタリング
        for result in processing_reading_results:
            if result and result.is_related:
                reading_results.append(result)

        # デバッグ用ログ
        if reading_results:
            logger.info("======== reading_results有 organize_results ========")
        else:
            logger.info("======== reading_results無し organize_results ========")
        return {"reading_results": reading_results}


graph = PaperSearchAgent(
    settings.fast_llm,
    ArxivSearcher(settings.fast_llm)
).graph