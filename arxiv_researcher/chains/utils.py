from pathlib import Path

# ファイル名を受け取りそのファイルの内容を文字列で返却する関数
def load_prompt(name: str) -> str:
    prompt_path = Path(__file__).resolve().parent / "prompts" / f"{name}.prompt"
    return prompt_path.read_text(encoding="utf-8").strip()