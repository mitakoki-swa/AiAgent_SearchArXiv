from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate

from arxiv_researcher.settings import settings
from arxiv_researcher.chains.task_evaluator_chain import TaskEvaluation
from arxiv_researcher.chains.utils import load_prompt

class DecomposedTasks(BaseModel):
    tasks: list[str] = Field(
        default_factory=list,
        min_length=settings.query_decomposer.min_decomposed_tasks,
        max_length=settings.query_decomposer.max_decomposed_tasks,
        description="分解されたタスクのリスト"
    )


class QueryDecomposer:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.min_decomposed_tasks = settings.query_decomposer.min_decomposed_tasks
        self.max_decomposed_tasks = settings.query_decomposer.max_decomposed_tasks

    def __call__(self, state: dict) -> Command[Literal["paper_search_agent"]]:
        evaluation: TaskEvaluation | None = state.get("evaluation", None)
        content = evaluation.content if evaluation else state.get("goal", "")
        decomposed_tasks: DecomposedTasks = self.run(content)
        return Command(
            goto="paper_search_agent",
            update={"tasks": decomposed_tasks.tasks}
        )

    def run(self, query: str) -> DecomposedTasks:
        prompt = ChatPromptTemplate.from_template(load_prompt("query_decomposer"))
        chain = prompt | self.llm.with_structured_output(
            DecomposedTasks,
            method="function_calling"
        )
        return chain.with_retry().invoke(
            {
                "min_decomposed_tasks": self.min_decomposed_tasks,
                "max_decomposed_tasks": self.max_decomposed_tasks,
                "current_date": self.current_date,
                "query": query
            }
        )