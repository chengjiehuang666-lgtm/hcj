"""安全基线（W9）：输入过滤、输出 DLP 脱敏、API Key 角色权限。

三个子模块互不依赖业务代码，便于单独测试：

- `input_guard`：Prompt Injection 规则 + 敏感词过滤，PII/密钥值就地脱敏；
- `dlp`：输出侧正则脱敏（身份证 / 手机号 / 银行卡 / 密钥）；
- `auth`：单租户 API Key → 简单角色（admin / operator / viewer）。
"""

from .auth import PERMISSIONS, Principal, authenticate, parse_api_keys
from .dlp import redact_structure, redact_text
from .input_guard import InputVerdict, scan_input
from .rate_limit import RateLimiter

__all__ = [
    "PERMISSIONS",
    "Principal",
    "RateLimiter",
    "authenticate",
    "parse_api_keys",
    "redact_structure",
    "redact_text",
    "InputVerdict",
    "scan_input",
]
