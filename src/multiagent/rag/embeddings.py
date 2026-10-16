"""本地 embedding：fastembed（ONNX 运行时）+ bge 系列模型。

**为什么用本地模型而不是 embedding API**：
- 检索每次任务都要用，走本地没有按次成本，也不受供应商可用性影响；
- 本机可用的智谱官方 embedding 端点需要余额，本地模型今天就能用。

**为什么是 bge-small-zh-v1.5 而不是架构文档写的 bge-m3**：
bge-m3（1024 维）模型文件约 2.2GB，而本机 C 盘仅剩 2.8G；bge-small-zh-v1.5
只有约 95MB / 512 维，对本项目的语料规模足够。这与「MVP 用 pgvector、规模化再上
Milvus」是同一类取舍。换模型需同步改：
`knowledge_chunks.embedding` 的维度（vector(512)）并重新灌库。

**为什么不直接 import fastembed 在模块顶层**：它会连带加载 onnxruntime，拖慢
所有导入本包的路径（包括只跑 SQL 沙箱的测试）。故惰性加载。
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

from ..config import Settings

if TYPE_CHECKING:  # pragma: no cover
    from fastembed import TextEmbedding


class EmbeddingError(Exception):
    """embedding 模型加载或推理失败。"""


class LocalEmbedder:
    """对 fastembed 的薄封装：惰性加载、批量编码、维度自检。"""

    def __init__(self, settings: Settings) -> None:
        self._model_name = settings.embedding_model
        self._cache_dir = settings.embedding_cache_dir
        self._hf_endpoint = settings.hf_endpoint
        self._expected_dim = settings.embedding_dim
        self._model: TextEmbedding | None = None
        self._actual_dim: int | None = None
        # 检索现在跑在线程里（见 retriever 的 asyncio.to_thread），而 embedder 是
        # 进程内共享的 → 可能被多个线程同时调用。onnxruntime 的会话本身线程安全，
        # 但 fastembed 的封装未作此保证，加锁是最便宜的保险；编码本身只有毫秒级。
        self._lock = threading.Lock()

    # ---- 模型加载 ----
    def _ensure_model(self) -> "TextEmbedding":
        if self._model is not None:
            return self._model

        # 必须在加载模型**之前**设置：直连 HuggingFace 在本网络不可达
        if self._hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", self._hf_endpoint)
        # 关闭软链接（Windows 非开发者模式不支持），并压掉相应告警
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError(
                "未安装 fastembed。请执行：pip install -e \".[rag]\""
            ) from exc

        try:
            self._model = TextEmbedding(
                model_name=self._model_name, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise EmbeddingError(
                f"加载 embedding 模型 {self._model_name} 失败"
                f"（缓存目录 {self._cache_dir}，镜像 {self._hf_endpoint}）: {exc}"
            ) from exc
        return self._model

    # ---- 对外 ----
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        """实际维度。首次调用会触发模型加载。"""
        if self._actual_dim is None:
            vectors = list(self._ensure_model().embed(["维度自检"]))
            self._actual_dim = len(vectors[0])
            if self._actual_dim != self._expected_dim:
                raise EmbeddingError(
                    f"模型 {self._model_name} 实际维度 {self._actual_dim} "
                    f"与配置 embedding_dim={self._expected_dim} 不一致；"
                    f"需同步修改 knowledge_chunks.embedding 的 vector(...) 并重新灌库。"
                )
        return self._actual_dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with self._lock:  # 见 __init__ 里关于线程安全的说明
                vectors = list(self._ensure_model().embed(texts))
        except Exception as exc:
            raise EmbeddingError(f"生成向量失败: {exc}") from exc
        out = [[float(x) for x in vec] for vec in vectors]
        if self._actual_dim is None:
            self._actual_dim = len(out[0])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


def to_pgvector_literal(vector: list[float]) -> str:
    """转成 pgvector 的字面量 `[a,b,c]`，供 `%s::vector` 使用。"""
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


# 进程内共享的 embedder。
# 为什么需要：加载模型要 1–2 秒，而一次评测会跑 20 个任务；
# 每个任务新建一个 embedder 会让模型反复加载，白白拖长总耗时。
_EMBEDDER_CACHE: dict[tuple[str, str], LocalEmbedder] = {}


def get_shared_embedder(settings: Settings) -> LocalEmbedder:
    """按 (模型, 缓存目录) 复用 embedder；同进程内模型只加载一次。"""
    key = (settings.embedding_model, settings.embedding_cache_dir)
    embedder = _EMBEDDER_CACHE.get(key)
    if embedder is None:
        embedder = LocalEmbedder(settings)
        _EMBEDDER_CACHE[key] = embedder
    return embedder
