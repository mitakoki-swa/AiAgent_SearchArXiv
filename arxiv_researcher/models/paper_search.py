from typing_extensions import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from arxiv_researcher.chains.hearing_chain import Hearing
from arxiv_researcher.chains.task_evaluator_chain import TaskEvaluation

from arxiv_researcher.models.reading import ReadingResult


class SearchAgentInputState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class SearchAgentOutputState(TypedDict):
    final_output: str

class SearchAgentProcessState(TypedDict):
    llm_response: Hearing
    goal: str
    retry_count: int
    tasks: list[str]
    evaluation: TaskEvaluation
    reading_results: list[ReadingResult]

class SearchAgentState(
    SearchAgentInputState,
    SearchAgentOutputState,
    SearchAgentProcessState
):
    pass