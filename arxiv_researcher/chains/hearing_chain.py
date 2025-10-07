from datetime import datetime
from typing_extensions import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from arxiv_researcher.chains.utils import load_prompt
from arxiv_researcher.chains.utils import format_history

# 内部ステート
class Hearing(BaseModel):
    """user_hearingのchain部分
    """
    is_need_human_feedback: bool = Field(
        default=False, description="追加質問が必要かどうか"
    )
    additional_question: str = Field(
        default=None, description="追加の質問"
    )

# user_hearingにおけるchain
class HearingChain:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.current_date = datetime.now().strftime("%Y-%m-%d")


    def __call__(self, state: dict) -> Command[Literal["human_feedback", "goal_setting"]]:
        '''テスト用
        goal_setting → goal_setting に変更して動かしている
        '''
        messages = state.get("messages", [])
        # hearing → llm_responseに変更
        llm_response = self.run(messages)
        message = []

        if llm_response.is_need_human_feedback:
            message = [{"role": "assistant", "content": llm_response.additional_question}]

        next_node = "human_feedback" if llm_response.is_need_human_feedback else "goal_setting"

        return Command(
            goto=next_node,
            update={"llm_response": llm_response, "messages": message}
        )

    def run(self, messages: list[BaseMessage]) -> Hearing:
        try:
            prompt = ChatPromptTemplate.from_template(load_prompt("hearing"))
            chain = prompt | self.llm.with_structured_output(
                Hearing,
                method="function_calling"
            )
            llm_response = chain.invoke({
                "current_date": self.current_date,
                "conversation_history": format_history(messages)
            })
        except Exception as e:
            raise RuntimeError(f"LLMの呼び出し中にエラーが発生しました: {str(e)}")

        return llm_response