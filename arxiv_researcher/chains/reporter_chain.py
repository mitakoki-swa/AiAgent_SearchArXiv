from datetime import datetime

from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from arxiv_researcher.chains.utils import load_prompt, dict_to_xml_str
from arxiv_researcher.models.reading import ReadingResult

class Reporter:
    def __init__(self, llm: ChatOpenAI) -> None:
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")

    def __call__(self, state: dict) -> Command:
        results: list[ReadingResult] = state["reading_results"]
        query: str = state["goal"]
        final_output: str = self.run(
            context="\n".join(
                [
                    dict_to_xml_str(item.model_dump(), exclude_keys=["markdown_text"])
                    for item in results
                ]
            ),
            query=query
        )
        return Command(update={"final_output": final_output})

    def run(self, context: str, query: str) -> str:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", load_prompt("reporter_system")),
                ("user", load_prompt("reporter_user"))
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke(
            {
                "current_date": self.current_date,
                "context": context,
                "query": query
            }
        )