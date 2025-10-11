FROM python:3.11-slim
WORKDIR /app

# 依存を先に入れてキャッシュを効かせる
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリ本体
COPY . .

# 公開ポート
EXPOSE 2024

# inmemならDB不要。ホスト公開を忘れずに
CMD ["uv", "run", "langgraph", "dev", "--config", "langgraph.json", "--host", "0.0.0.0", "--port", "2024"]