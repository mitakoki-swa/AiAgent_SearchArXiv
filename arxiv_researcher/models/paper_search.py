from typing import Annotated, TypedDict

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

from arxiv_researcher.chains.hearing_chain import Hearing

"""
下記は本番環境にてresearch_agent.py内に定義されていた。
それをmodelsに移している。
"""

class SearchAgentInputState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class SearchAgentOutputState(TypedDict):
    final_output: str

class SearchAgentProcessState(TypedDict):
    #llm_response: Hearing
    pass

class SearchAgentState(
    SearchAgentInputState,
    SearchAgentOutputState,
    SearchAgentProcessState
):
    pass