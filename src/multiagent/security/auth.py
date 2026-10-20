"""单租户 API Key + 简单角色权限（W9）。

角色只有三个，权限是**固定的映射**，不做 RBAC/ABAC 模型：

| 角色 | tasks:read | tasks:submit | tasks:approve |
| :--- | :---: | :---: | :---: |
| `viewer` | ✅ | ❌ | ❌ |
| `operator` | ✅ | ✅ | ❌ |
| `admin` | ✅ | ✅ | ✅ |

审批（HITL）是高风险动作，只有 admin 能做；提交与读取分给 operator/viewer。

Key 配置格式：`API_KEYS=key1:admin,key2:operator,key3:viewer`。
解析时按**最后一个冒号**切分角色（Key 里允许有冒号）；重复 Key、未知角色、
空 Key 都直接报错——配置错了应当启动失败，而不是静默放行。
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass

PERMISSIONS: dict[str, frozenset[str]] = {
    "admin": frozenset({"tasks:read", "tasks:submit", "tasks:approve"}),
    "operator": frozenset({"tasks:read", "tasks:submit"}),
    "viewer": frozenset({"tasks:read"}),
}


@dataclass(frozen=True)
class Principal:
    """已认证的调用方。**不保存原始 Key**，只留指纹供日志/审计。"""

    key_id: str
    role: str

    def can(self, permission: str) -> bool:
        return permission in PERMISSIONS.get(self.role, frozenset())


def fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def parse_api_keys(raw: str) -> dict[str, str]:
    """解析 `key:role,key:role`；返回 {key: role}。"""
    result: dict[str, str] = {}
    for item in raw.replace(";", ",").replace("\n", ",").split(","):
        item = item.strip()
        if not item:
            continue
        key, sep, role = item.rpartition(":")
        key = key.strip()
        role = role.strip().lower()
        if not sep or not key:
            raise ValueError(f"API_KEYS 条目格式应为 key:role，实际为 {item!r}")
        if role not in PERMISSIONS:
            raise ValueError(
                f"API_KEYS 中的角色 {role!r} 无效（可选：{', '.join(PERMISSIONS)}）"
            )
        if key in result:
            raise ValueError("API_KEYS 中存在重复 Key")
        result[key] = role
    return result


def authenticate(token: str | None, keys: dict[str, str]) -> Principal | None:
    """常量时间比较所有 Key；命中返回 Principal，否则 None。"""
    if not token:
        return None
    found: str | None = None
    for key, role in keys.items():
        if secrets.compare_digest(token, key):
            found = role
    if found is None:
        return None
    return Principal(key_id=fingerprint(token), role=found)
