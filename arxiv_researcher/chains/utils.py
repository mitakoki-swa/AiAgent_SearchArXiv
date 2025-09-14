from pathlib import Path

from langchain_core.messages import BaseMessage

# ファイル名を受け取りそのファイルの内容を文字列で返却する関数
def load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / f"{name}.prompt"
    return prompt_path.read_text(encoding="utf-8").strip()

# list[BaseMessage]をstrに変換する関数
def format_history(messages: list[BaseMessage]) -> str:
    return "\n".join([f"{message.type}: {message.content}" for message in messages])

# pydanticスキーマをxml形式のstrに変換する関数
def dict_to_xml_str(data: dict, exclude_keys: list[str] = []) -> str:
    xml_str = "<item>"
    for key, value in data.items():
        if key not in exclude_keys:
            xml_str += f"<{key}>{value}</{key}>"
    xml_str += "</item>"
    return xml_str