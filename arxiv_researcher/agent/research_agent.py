from typing import Annotated, Literal, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from langchain_core.messages import HumanMessage

from arxiv_researcher.models.paper_search import SearchAgentState, SearchAgentInputState, SearchAgentOutputState
from arxiv_researcher.chains.hearing_chain import HearingChain
from arxiv_researcher.chains.reporter_chain import Reporter


class ResearchAgent:
    def __init__(
            self,
            llm: ChatOpenAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.0),
    ) -> None:
        self.recursion_limit = 1000
        self.max_evaluation_retry_count = 3
        self.user_hearing = HearingChain(llm)
        self.generate_report = Reporter(llm)
        self.graph = self._create_graph()

    def _create_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(
            state_schema=SearchAgentState,
            input_schema=SearchAgentInputState,
            output_schema=SearchAgentOutputState
        )
        workflow.add_node("user_hearing", self.user_hearing)
        workflow.add_node("human_feedback", self._human_feedback)
        workflow.add_node("generate_report", self.generate_report)

        workflow.set_entry_point("user_hearing")
        workflow.set_finish_point("generate_report")

        return workflow.compile()
    
    def _human_feedback(self, state: SearchAgentState) -> Command[Literal["user_hearing"]]:
        # 最後のメッセージ取得
        last_message = state["messages"][-1]
        # ユーザーへの質問表示
        human_feedback = interrupt(last_message.content)
        if human_feedback is None or human_feedback == "Empty message":
            human_feedback = "そのままの条件で検索し、調査してください"
        return Command(
            goto="user_hearing",
            update={"messages": [HumanMessage(content=human_feedback)]}
        )



graph = ResearchAgent().graph

