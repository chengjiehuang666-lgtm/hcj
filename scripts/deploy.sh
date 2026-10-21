#!/usr/bin/env bash
# W12/W14 单机部署：依赖服务 + 应用容器；可选 HTTPS（自签证书）。
# 用法：
#   bash scripts/deploy.sh          # HTTP，仅本机 127.0.0.1:8000
#   bash scripts/deploy.sh --tls    # HTTPS 8443（自签证书）+ 本机 HTTP 8000
set -euo pipefail

cd "$(dirname "$0")/.."

TLS=0
for arg in "$@"; do
  case "$arg" in
    --tls) TLS=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

if [ ! -f .env ]; then
  echo "缺少 .env：先 cp .env.example .env 并填好 Key（至少 API_KEYS）" >&2
  exit 1
fi
mkdir -p .model_cache audit

TAG="$(git describe --tags --always --dirty 2>/dev/null || echo local)"
IMAGE="multiagent-app:${TAG}"
COMPOSE=(-f docker-compose.yml -f docker-compose.app.yml)
HEALTH_URL="http://127.0.0.1:8000/health"
CURL_OPTS=(-sf -m 5)
SMOKE_BASE="http://127.0.0.1:8000"

if [ "$TLS" -eq 1 ]; then
  bash scripts/gen_cert.sh
  COMPOSE+=(-f docker-compose.tls.yml)
  HEALTH_URL="https://127.0.0.1:8443/health"
  CURL_OPTS+=(-k)
  SMOKE_BASE="https://127.0.0.1:8443"
  echo "▶ TLS 模式：https://localhost:8443（自签证书，浏览器需信任/忽略告警）"
fi

echo "▶ 构建并启动（镜像 ${IMAGE}）"
APP_IMAGE="$IMAGE" docker compose "${COMPOSE[@]}" up -d --build

echo "▶ 等待 ${HEALTH_URL}"
for _ in $(seq 1 90); do
  if curl "${CURL_OPTS[@]}" "$HEALTH_URL" >/dev/null; then
    break
  fi
  sleep 2
done
if ! curl "${CURL_OPTS[@]}" "$HEALTH_URL" >/dev/null; then
  echo "✗ 服务未在 180 秒内就绪；查看：docker compose ${COMPOSE[*]} logs app" >&2
  exit 1
fi
echo "PASS health"

PY="python"
if [ -x .venv/Scripts/python.exe ]; then
  PY=".venv/Scripts/python.exe"
elif [ -x .venv/bin/python ]; then
  PY=".venv/bin/python"
fi
SMOKE_BASE_URL="$SMOKE_BASE" SMOKE_INSECURE="$TLS" "$PY" scripts/smoke.py

STATE="$TAG"
[ "$TLS" -eq 1 ] && STATE="${TAG}+tls"
printf '%s\n' "$STATE" > .deploy-state
echo "✅ 部署完成：${STATE}（记录在 .deploy-state）"
if [ "$TLS" -eq 1 ]; then
  echo "   控制台：https://localhost:8443/console/（自签证书）"
else
  echo "   控制台：http://localhost:8000/console/"
fi
echo "   日志：docker compose ${COMPOSE[*]} logs -f app"
echo "   审计：audit/audit.jsonl；指标：curl -H 'Authorization: Bearer <key>' ${SMOKE_BASE}/metrics"
