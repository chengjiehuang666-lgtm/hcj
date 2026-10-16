-- ============================================================================
-- 知识库（RAG）表结构。
--
-- 为什么建在 initdb 而不是应用启动时：表结构属于数据库层，与业务数据分离更清晰；
-- 数据（分块 + 向量）由应用侧写入，见 src/multiagent/rag/ingest.py。
--
-- 向量维度必须与 embedding 模型一致：当前用 BAAI/bge-small-zh-v1.5（512 维）。
-- 换了 embedding 模型就要改这里的 vector(512) 并重新灌库。
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id          bigserial PRIMARY KEY,
    doc_id      text        NOT NULL,
    chunk_index integer     NOT NULL,
    title       text        NOT NULL,
    content     text        NOT NULL,
    -- embedding 允许为空：分块先入库、向量可后补，
    -- 也让「关键词检索」在向量缺失时仍可用。
    embedding   vector(512),
    created_at  timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT knowledge_chunks_unique UNIQUE (doc_id, chunk_index)
);

COMMENT ON TABLE knowledge_chunks IS '知识库分块。向量由应用侧 fastembed 生成（bge-small-zh-v1.5，512 维）。';
COMMENT ON COLUMN knowledge_chunks.embedding IS '归一化前的原始向量；检索时用余弦距离（<=>）。';

-- 向量近邻索引。语料很小时顺序扫描也够用，但先把索引建好，规模化时不必改代码。
CREATE INDEX IF NOT EXISTS idx_knowledge_embedding
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);

-- 关键词检索（pg_trgm）用的索引，供混合检索的另一路使用
CREATE INDEX IF NOT EXISTS idx_knowledge_content_trgm
    ON knowledge_chunks USING gin (content gin_trgm_ops);

-- 只读角色要能检索知识库（Agent 的检索走 analyst_ro）
GRANT SELECT ON knowledge_chunks TO analyst_ro;
GRANT SELECT ON SEQUENCE knowledge_chunks_id_seq TO analyst_ro;
