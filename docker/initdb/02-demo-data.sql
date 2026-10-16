-- ============================================================================
-- 阶段一 MVP 演示数据 + SQL 沙箱的只读角色
--
-- 仅在数据卷为空时执行一次。用途：
--   1) 给「数据分析流水线」场景提供可分析的业务数据；
--   2) 建立一个**只读**数据库角色，作为 Agent 执行 SQL 的安全边界
--      （见《实施路线图》W3「SQL 沙箱」与《系统架构设计》2.2.5）。
--
-- 数据是刻意构造的演示数据，不是真实业务数据。
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 业务表：销售明细
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sales (
    id            bigserial PRIMARY KEY,
    stat_date     date          NOT NULL,
    region        text          NOT NULL,
    channel       text          NOT NULL,
    category      text          NOT NULL,
    revenue       numeric(14,2) NOT NULL,
    orders        integer       NOT NULL,
    refund_amount numeric(14,2) NOT NULL DEFAULT 0,
    CONSTRAINT sales_nonneg CHECK (revenue >= 0 AND orders >= 0 AND refund_amount >= 0)
);

COMMENT ON TABLE sales IS '销售明细（演示数据）。revenue 为销售额，refund_amount 为退货金额。';
CREATE INDEX IF NOT EXISTS idx_sales_date_region ON sales (stat_date, region);

-- ---------------------------------------------------------------------------
-- 指标口径表：供检索 Agent 查询「口径定义」，避免分析口径不一致
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS metric_definitions (
    metric_key       text PRIMARY KEY,
    metric_name      text NOT NULL,
    definition       text NOT NULL,
    formula          text NOT NULL,
    owner_department text NOT NULL
);

COMMENT ON TABLE metric_definitions IS '指标口径库。分析前必须先取口径，保证结论可解释。';

INSERT INTO metric_definitions (metric_key, metric_name, definition, formula, owner_department) VALUES
    ('revenue',      '销售额',   '统计周期内已完成支付的订单金额合计，不含已全额退款订单。',
     'SUM(revenue)', '财务部'),
    ('orders',       '订单量',   '统计周期内创建且支付成功的订单数，不含取消订单。',
     'SUM(orders)', '运营部'),
    ('refund_amount','退货金额', '统计周期内实际发生的退款金额合计。',
     'SUM(refund_amount)', '售后部'),
    ('refund_rate',  '退货率',   '退货金额占同期销售额的比例，用于衡量销售质量。',
     'SUM(refund_amount) / NULLIF(SUM(revenue), 0)', '售后部'),
    ('region_east',  '华东区',   '指上海、江苏、浙江、安徽、福建、江西、山东七省市。',
     'region = ''华东''', '运营部'),
    ('channel',      '渠道',     '销售渠道分为「线上」与「门店」两类，口径互斥不重叠。',
     'channel IN (''线上'', ''门店'')', '运营部')
ON CONFLICT (metric_key) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 演示数据：华东区 2026-09 销售额下滑，华南区基本持平
-- 故事线：华东区下滑主因是「线上 × 数码」——订单量下降且退货额翻倍
-- ---------------------------------------------------------------------------
INSERT INTO sales (stat_date, region, channel, category, revenue, orders, refund_amount) VALUES
    -- 华东区
    ('2026-07-01','华东','线上','数码', 1200000, 5000,  58000),
    ('2026-07-01','华东','线上','家居',  400000, 2100,  22000),
    ('2026-07-01','华东','门店','数码',  900000, 2400,  30000),
    ('2026-07-01','华东','门店','家居',  350000, 1500,  15000),
    ('2026-08-01','华东','线上','数码', 1250000, 5200,  60000),
    ('2026-08-01','华东','线上','家居',  420000, 2200,  23000),
    ('2026-08-01','华东','门店','数码',  920000, 2450,  31000),
    ('2026-08-01','华东','门店','家居',  360000, 1550,  16000),
    ('2026-09-01','华东','线上','数码',  780000, 3300, 145000),
    ('2026-09-01','华东','线上','家居',  300000, 1600,  41000),
    ('2026-09-01','华东','门店','数码',  900000, 2400,  32000),
    ('2026-09-01','华东','门店','家居',  355000, 1530,  16500),
    -- 华南区（对照组：基本持平）
    ('2026-07-01','华南','线上','数码',  700000, 2900,  33000),
    ('2026-07-01','华南','线上','家居',  260000, 1300,  14000),
    ('2026-07-01','华南','门店','数码',  560000, 1500,  19000),
    ('2026-07-01','华南','门店','家居',  240000, 1000,  10000),
    ('2026-08-01','华南','线上','数码',  720000, 3000,  34000),
    ('2026-08-01','华南','线上','家居',  265000, 1320,  14500),
    ('2026-08-01','华南','门店','数码',  575000, 1540,  19500),
    ('2026-08-01','华南','门店','家居',  245000, 1020,  10200),
    ('2026-09-01','华南','线上','数码',  715000, 2980,  34500),
    ('2026-09-01','华南','线上','家居',  263000, 1310,  14400),
    ('2026-09-01','华南','门店','数码',  573000, 1535,  19400),
    ('2026-09-01','华南','门店','家居',  243000, 1010,  10150)
ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------------------
-- SQL 沙箱的只读角色
-- 只授予 SELECT；不授予 INSERT/UPDATE/DELETE；禁止建表；语句超时兜底。
-- Agent 执行查询一律使用本角色，不使用应用主账号。
-- ---------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'analyst_ro') THEN
        CREATE ROLE analyst_ro WITH LOGIN PASSWORD 'analyst_ro';
    END IF;
END $$;

GRANT CONNECT ON DATABASE multiagent TO analyst_ro;
GRANT USAGE ON SCHEMA public TO analyst_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO analyst_ro;

-- 显式回收写权限，双保险
REVOKE CREATE ON SCHEMA public FROM analyst_ro;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON ALL TABLES IN SCHEMA public FROM analyst_ro;

-- 语句级超时，防止 Agent 生成慢查询拖垮数据库
ALTER ROLE analyst_ro SET statement_timeout = '5s';
ALTER ROLE analyst_ro SET idle_in_transaction_session_timeout = '10s';
