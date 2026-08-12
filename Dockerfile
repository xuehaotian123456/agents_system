# Agent 双引擎系统 — Docker 镜像
# 构建: docker build -t agent-system .
# 运行: docker-compose up

FROM python:3.11-slim

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY pipeline/requirements.txt /app/pipeline/
RUN pip install --no-cache-dir -r /app/pipeline/requirements.txt

COPY interaction/requirements.txt /app/interaction/
RUN pip install --no-cache-dir -r /app/interaction/requirements.txt

RUN pip install --no-cache-dir apscheduler pyvis jieba chromadb feedparser httpx beautifulsoup4

# 代码
COPY pipeline/ /app/pipeline/
COPY interaction/ /app/interaction/

# 数据目录
RUN mkdir -p /app/pipeline/data/articles /app/pipeline/data/graphs /app/pipeline/cache

# 端口
EXPOSE 8010 8020

# 默认启动 Pipeline A2A（可 override）
CMD ["python", "-m", "uvicorn", "pipeline.a2a_server:app", "--host", "0.0.0.0", "--port", "8010"]
