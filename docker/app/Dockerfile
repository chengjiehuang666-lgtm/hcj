# 应用镜像（W12）：单机部署用；依赖服务见 docker-compose.yml。
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 仅装运行期依赖；agents/rag 是可选依赖组（LangGraph、fastembed/onnxruntime）
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[agents,rag]"

# 初始化 SQL 等部署资产（不包含密钥；.env 由 compose env_file 注入）
COPY docker ./docker

EXPOSE 8000

# 事件循环：uvicorn 在 Windows 上需 Selector，Linux 容器用默认即可
CMD ["python", "-m", "multiagent.api", "--host", "0.0.0.0", "--port", "8000"]
