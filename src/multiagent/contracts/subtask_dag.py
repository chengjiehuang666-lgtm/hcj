"""SubtaskDAG —— 对应 `架构设想/接口契约/subtask_dag.schema.json`。

Planner 输出的子任务图。

契约声明 JSON Schema 无法表达无环性，要求 Orchestrator 额外校验三件事。
本模块把这三项校验前置到构造期：**构造即校验**，非法 DAG 无法被创建，
避免带缺陷的图流到调度层。
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentRole(str, Enum):
    """可执行节点的 Agent 角色。

    注意：planner 是本图的**产出者**而非节点，故不在枚举内。
    """

    retriever = "retriever"
    analyst = "analyst"
    reviewer = "reviewer"
    writer = "writer"


class SubtaskNode(_ContractModel):
    node_id: str = Field(pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1)
    agent_role: AgentRole
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    acceptance: list[str] = Field(min_length=1)
    requires_human_approval: bool = False
    timeout_seconds: int = Field(default=300, ge=1)
    max_retries: int = Field(default=2, ge=0)


class SubtaskDAG(_ContractModel):
    task_id: str
    schema_version: Literal["1.0.0"] = "1.0.0"
    goal: str = Field(min_length=1)
    nodes: list[SubtaskNode] = Field(min_length=1)
    created_at: datetime
    revision: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_graph(self) -> SubtaskDAG:
        ids = [n.node_id for n in self.nodes]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"node_id 重复: {dupes}")
        known = set(ids)

        # 校验 2：依赖引用必须存在
        for node in self.nodes:
            for dep in node.depends_on:
                if dep not in known:
                    raise ValueError(f"节点 '{node.node_id}' 依赖不存在的节点 '{dep}'")

        # 校验 1：无环（DFS 三色标记）
        by_id = {n.node_id: n for n in self.nodes}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(ids, WHITE)

        def visit(nid: str, path: list[str]) -> None:
            color[nid] = GRAY
            for dep in by_id[nid].depends_on:
                if color[dep] == GRAY:
                    cycle = " -> ".join([*path, nid, dep])
                    raise ValueError(f"检测到循环依赖: {cycle}")
                if color[dep] == WHITE:
                    visit(dep, [*path, nid])
            color[nid] = BLACK

        for nid in ids:
            if color[nid] == WHITE:
                visit(nid, [])

        # 校验 3：所有节点必须从某个入口可达（入口 = 无依赖的节点）
        # 注意遍历方向：depends_on 指向"上游"，从入口出发必须沿"下游"（依赖它的节点）走。
        roots = [n.node_id for n in self.nodes if not n.depends_on]
        if not roots:
            raise ValueError("不存在入口节点（所有节点都有依赖）")

        dependents: dict[str, list[str]] = {nid: [] for nid in ids}
        for node in self.nodes:
            for dep in node.depends_on:
                dependents[dep].append(node.node_id)

        reachable: set[str] = set()
        stack = list(roots)
        while stack:
            nid = stack.pop()
            if nid in reachable:
                continue
            reachable.add(nid)
            stack.extend(dependents[nid])
        if unreachable := known - reachable:
            raise ValueError(f"存在从入口不可达的节点: {sorted(unreachable)}")

        return self

    def ready_nodes(self, completed: set[str]) -> list[SubtaskNode]:
        """返回依赖已全部完成、可以立即调度的节点。"""
        return [
            n
            for n in self.nodes
            if n.node_id not in completed and set(n.depends_on) <= completed
        ]
