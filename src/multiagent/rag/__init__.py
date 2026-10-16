"""RAG 层：本地 embedding + 分块 + pgvector 混合检索。

设计要点见各模块 docstring；关键的取舍是「本地小模型优先」（见 embeddings.py）。
"""

from .chunking import Chunk, chunk_document
from .corpus import DOCUMENTS
from .embeddings import (
    EmbeddingError,
    LocalEmbedder,
    get_shared_embedder,
    to_pgvector_literal,
)
from .store import Document, KnowledgeStore, KnowledgeStoreError, RetrievedChunk

__all__ = [
    "LocalEmbedder",
    "get_shared_embedder",
    "EmbeddingError",
    "to_pgvector_literal",
    "KnowledgeStore",
    "KnowledgeStoreError",
    "RetrievedChunk",
    "Document",
    "Chunk",
    "chunk_document",
    "DOCUMENTS",
]
