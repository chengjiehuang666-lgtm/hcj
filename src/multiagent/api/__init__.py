"""交互入口层：REST 提交任务 + SSE 订阅进度。"""

from .app import create_app
from .tasks import TaskRecord, TaskRegistry

__all__ = ["create_app", "TaskRegistry", "TaskRecord"]
