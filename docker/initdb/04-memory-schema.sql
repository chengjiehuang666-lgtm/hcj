-- 长期记忆（W6）：跨会话沉淀的结论。
--
-- 为什么**不复用 knowledge_chunks**：
--   1) 语义与生命周期不同——知识库是外部文档（灌库即静态），记忆是任务产物
--      （可删除/更正、带 TTL、有版本流转）；
--   2) 评测的 RAG 回归用例按「检索到的 knowledge 文档」做结构性断言，
--      记忆若混入同一张表/同一个检索出口，那些断言会被污染、静默失效。
--
-- 检索口径与知识库一致：向量（bge-small-zh-v1.5，512 维）+ pg_trgm 关键词线性加权。
--
-- 注意：initdb 只在**数据卷为空时**执行一次；对已有环境需手工执行本文件：
--   docker exec -i ma-postgres psql -U multiagent -d multiagent < docker/initdb/04-memory-schema.sql

CREATE TABLE IF NOT EXISTS memory_entries (
    id            bigserial PRIMARY KEY,
    task_id       text        NOT NULL,
    node_id       text        NOT NULL,
    kind          text        NOT NULL DEFAULT 'conclusion',  -- conclusion | fact | preference
    content       text        NOT NULL,
    embedding     vector(512),                                -- 可空：关键词路在向量缺失时仍可用
    source_type   text        NOT NULL,                       -- 复用 SourceType 取值
    source_ref    text        NOT NULL,                       -- 来源出处（任务/节点/SQL 哈希）
    confidence    real,
    status        text        NOT NULL DEFAULT 'draft',       -- draft | validated | rejected | superseded
    superseded_by bigint REFERENCES memory_entries(id),
    ttl_seconds   integer,
    created_at    timestamptz NOT NULL DEFAULT now(),
    -- 同一任务不重复写入（task_id 每次运行都是新 UUID，仅此约束**防不了**
    -- 同题重跑堆重复结论——跨运行去重在应用层按 content 判重，见 memory/store.py）
    CONSTRAINT memory_entries_task_unique UNIQUE (task_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_embedding
    ON memory_entries USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_memory_content_trgm
    ON memory_entries USING gin (content gin_trgm_ops);

-- 与知识库同权限口径：检索走只读角色（02-demo-data.sql 的默认权限已覆盖新建表，
-- 这里显式声明以免默认权限被改动后静默失效）
GRANT SELECT ON memory_entries TO analyst_ro;
GRANT SELECT ON SEQUENCE memory_entries_id_seq TO analyst_ro;
