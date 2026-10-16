"""编排层：LangGraph 图状态机与执行入口。

阶段一 W2 为**固定串行链路** Planner → Retriever → Analyst；
动态 DAG 调度（按 SubtaskDAG 并发推进）在 W5 落地。
"""

from .pipeline import AnalysisPipeline, PipelineNodeFailed, PipelineOutcome

__all__ = ["AnalysisPipeline", "PipelineOutcome", "PipelineNodeFailed"]
