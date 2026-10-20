#!/usr/bin/env bash
# W12 单机部署：依赖服务 + 应用容器，构建后做健康检查与冒烟。
# 用法：bash scripts/deploy.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "缺少 .env：先 cp .env.example .env 并填好 Key（至少 API_KEYS）" >&2
  exit 1
fi
mkdir -p .model_cache

TAG="$(git describe --tags --always --dirty 2>/dev/null || echo local)"
IMAGE="multiagent-app:${TAG}"

echo "▶ 构建并启动（镜像 ${IMAGE}）"
APP_IMAGE="$IMAGE" docker compose \
  -f docker-compose.yml -f docker-compose.app.yml \
  up -d --build

echo "▶ 等待 /health"
for _ in $(seq 1 90); do
  if curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null; then
  echo "✗ 服务未在 180 秒内就绪；查看：docker compose -f docker-compose.yml -f docker-compose.app.yml logs app" >&2
  exit 1
fi
echo "PASS health"

PY="python"
if [ -x .venv/Scripts/python.exe ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
fi
"$PY" scripts/smoke.py

printf '%s\n' "$TAG" > .deploy-state
echo "✅ 部署完成：$TAG（记录在 .deploy-state）"
echo "   日志：docker compose -f docker-compose.yml -f docker-compose.app.yml logs -f app"
echo "   指标：curl -H 'Authorization: Bearer <key>' http://127.0.0.1:8000/metrics"
