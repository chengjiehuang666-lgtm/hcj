"""支持 `python -m multiagent.memory` 直接执行（转发给 cli.main）。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
