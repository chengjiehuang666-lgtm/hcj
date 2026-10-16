-- PostgreSQL 初始化脚本。
-- 仅在数据卷为空时由官方 entrypoint 执行一次，用于保证扩展可复现，
-- 避免每次重建环境都要手工敲 CREATE EXTENSION。

-- 向量检索：长期记忆与知识库（对应《系统架构设计》2.2.2「长期记忆」）
CREATE EXTENSION IF NOT EXISTS vector;

-- 模糊/关键词检索：与向量召回组成混合检索（BM25 的 PG 侧近似）
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- 任务 ID、Agent 版本等字符串的主键生成与短 ID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
