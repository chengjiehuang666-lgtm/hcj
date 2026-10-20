/* W13 轻量控制台：无构建链，直接用 REST / SSE / metrics。 */
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  key: sessionStorage.getItem("ma_api_key") || "",
  selected: null,
  es: null,
  timer: null,
};

const EVENTS = [
  "task_started", "task_resumed", "node_started", "node_finished", "node_failed",
  "blackboard_written", "memory_sunk", "memory_sink_failed", "task_breaker_tripped",
  "replan_triggered", "approval_required", "approval_received", "task_recovered",
  "input_blocked", "task_finished", "end",
];

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.classList.remove("hidden");
  clearTimeout(box._timer);
  box._timer = setTimeout(() => box.classList.add("hidden"), 5000);
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (state.key) headers.Authorization = `Bearer ${state.key}`;
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch (_) { body = text; }
  if (!response.ok) {
    if (response.status === 401) $("conn").className = "dot off";
    const detail = body && body.detail ? JSON.stringify(body.detail) : String(body || "");
    throw new Error(`HTTP ${response.status}: ${detail.slice(0, 300)}`);
  }
  return body;
}

const fmtInt = (n) => Number(n || 0).toLocaleString("zh-CN");
const fmtCost = (n) => Number(n || 0).toFixed(4);

function badge(status) {
  const span = document.createElement("span");
  span.className = `badge ${status || "pending"}`;
  span.textContent = status || "pending";
  return span;
}

function card(label, value) {
  const div = document.createElement("div");
  div.className = "card";
  const k = document.createElement("div"); k.className = "k"; k.textContent = label;
  const v = document.createElement("div"); v.className = "v"; v.textContent = value;
  div.append(k, v);
  return div;
}

async function checkHealth() {
  try {
    await api("/health");
    $("conn").className = "dot on";
  } catch (_) {
    $("conn").className = "dot off";
  }
}

async function loadTasks() {
  try {
    const body = await api("/tasks?limit=50");
    const tasks = body.tasks || [];
    $("task-count").textContent = `(${tasks.length})`;
    const list = $("task-list");
    list.replaceChildren();
    if (!tasks.length) {
      const empty = document.createElement("li");
      empty.className = "empty-item";
      empty.textContent = state.key ? "暂无任务" : "填写 API Key 后加载任务";
      list.append(empty);
    }
    for (const task of tasks) {
      const li = document.createElement("li");
      if (task.task_id === state.selected) li.classList.add("active");
      const q = document.createElement("div");
      q.className = "task-q"; q.textContent = task.question || "(无问题)"; q.title = task.question || "";
      const meta = document.createElement("div");
      meta.className = "task-meta";
      meta.append(badge(task.status));
      const id = document.createElement("span");
      id.textContent = (task.task_id || "").slice(0, 8);
      const at = document.createElement("span");
      at.textContent = (task.created_at || "").replace("T", " ").slice(0, 19);
      meta.append(id, at);
      li.append(q, meta);
      li.addEventListener("click", () => selectTask(task.task_id));
      list.append(li);
    }
  } catch (error) {
    toast(`加载任务失败：${error.message}`);
  }
}

function closeSSE() {
  if (state.es) { state.es.close(); state.es = null; }
}

function appendEvent(name, raw) {
  let data = {};
  try { data = JSON.parse(raw || "{}"); } catch (_) { data = { raw }; }
  const bits = [];
  for (const key of ["status", "node_id", "agent_role", "route", "error", "summary", "error_code"]) {
    if (data[key] !== undefined && data[key] !== null && data[key] !== "") {
      bits.push(`${key}=${String(data[key]).slice(0, 160)}`);
    }
  }
  const line = `[${new Date().toLocaleTimeString("zh-CN")}] ${name}${bits.length ? "  " + bits.join("  ") : ""}`;
  const box = $("events");
  box.textContent += (box.textContent ? "\n" : "") + line;
  box.scrollTop = box.scrollHeight;
}

function openSSE(taskId) {
  closeSSE();
  $("events").textContent = "";
  const query = state.key ? `?api_key=${encodeURIComponent(state.key)}` : "";
  const es = new EventSource(`/tasks/${taskId}/events${query}`);
  state.es = es;
  for (const name of EVENTS) {
    es.addEventListener(name, (event) => {
      appendEvent(name, event.data);
      if (name === "task_finished" || name === "approval_required") {
        loadTasks();
        if (state.selected === taskId) refreshSelected();
      }
      if (name === "end") closeSSE();
    });
  }
  es.onerror = () => {
    // 任务结束后 EventSource 会尝试重连；已收到 end 时不再提示
    if (state.es === es && es.readyState === EventSource.CLOSED) closeSSE();
  };
}

function renderDag(dag, nodeResults) {
  const box = $("dag");
  box.replaceChildren();
  const nodes = (dag && dag.nodes) || [];
  if (!nodes.length) {
    box.textContent = "（无 DAG 信息）";
    return;
  }
  const byId = new Map(nodes.map((n) => [n.node_id, n]));
  const depth = new Map();
  const calc = (id, seen) => {
    if (depth.has(id)) return depth.get(id);
    if (seen.has(id)) return 0;
    seen.add(id);
    const node = byId.get(id);
    const deps = (node && node.depends_on) || [];
    const value = deps.length ? Math.max(...deps.map((d) => calc(d, seen))) + 1 : 0;
    depth.set(id, value);
    return value;
  };
  nodes.forEach((n) => calc(n.node_id, new Set()));

  const layers = new Map();
  for (const node of nodes) {
    const level = depth.get(node.node_id) || 0;
    if (!layers.has(level)) layers.set(level, []);
    layers.get(level).push(node);
  }
  const maxLevel = Math.max(...layers.keys());
  const maxRows = Math.max(...[...layers.values()].map((items) => items.length));
  const NW = 150, NH = 50, GX = 64, GY = 22, PAD = 16;
  const width = PAD * 2 + (maxLevel + 1) * NW + maxLevel * GX;
  const height = PAD * 2 + maxRows * NH + (maxRows - 1) * GY;
  const pos = new Map();
  for (const [level, items] of layers) {
    const total = items.length * NH + (items.length - 1) * GY;
    const startY = PAD + (height - PAD * 2 - total) / 2;
    items.forEach((node, index) => {
      pos.set(node.node_id, {
        x: PAD + level * (NW + GX),
        y: startY + index * (NH + GY),
      });
    });
  }

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const defs = document.createElementNS(NS, "defs");
  const marker = document.createElementNS(NS, "marker");
  marker.setAttribute("id", "arrow");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "8");
  marker.setAttribute("refX", "7");
  marker.setAttribute("refY", "4");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS(NS, "path");
  arrowPath.setAttribute("d", "M0,0 L8,4 L0,8 z");
  arrowPath.setAttribute("fill", "#4c8dff");
  marker.append(arrowPath);
  defs.append(marker);
  svg.append(defs);

  const statusByNode = new Map((nodeResults || []).map((n) => [n.node_id, n.status]));
  for (const node of nodes) {
    for (const dep of node.depends_on || []) {
      const from = pos.get(dep), to = pos.get(node.node_id);
      if (!from || !to) continue;
      const path = document.createElementNS(NS, "path");
      const x1 = from.x + NW, y1 = from.y + NH / 2;
      const x2 = to.x, y2 = to.y + NH / 2;
      const mid = (x1 + x2) / 2;
      path.setAttribute("d", `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`);
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", "#3d5068");
      path.setAttribute("stroke-width", "1.5");
      path.setAttribute("marker-end", "url(#arrow)");
      svg.append(path);
    }
  }

  for (const node of nodes) {
    const p = pos.get(node.node_id);
    const status = statusByNode.get(node.node_id);
    const group = document.createElementNS(NS, "g");
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", String(p.x));
    rect.setAttribute("y", String(p.y));
    rect.setAttribute("width", String(NW));
    rect.setAttribute("height", String(NH));
    rect.setAttribute("rx", "8");
    rect.setAttribute("fill", "#1d2732");
    rect.setAttribute("stroke", status === "failed" ? "#f85149" : status === "ok" ? "#2ea043" : "#3d5068");
    group.append(rect);

    const title = document.createElementNS(NS, "text");
    title.setAttribute("x", String(p.x + 10));
    title.setAttribute("y", String(p.y + 20));
    title.textContent = node.node_id;
    const role = document.createElementNS(NS, "text");
    role.setAttribute("x", String(p.x + 10));
    role.setAttribute("y", String(p.y + 37));
    role.setAttribute("class", "role");
    role.textContent = `${node.agent_role || ""}${status ? " · " + status : ""}`;
    group.append(title, role);
    svg.append(group);
  }
  box.append(svg);
}

function renderDetail(body) {
  $("detail-empty").classList.add("hidden");
  $("detail-body").classList.remove("hidden");
  const taskId = body.task_id || "";
  const title = $("detail-title");
  title.replaceChildren(document.createTextNode(`任务 ${taskId.slice(0, 8)}  `), badge(body.status));

  const result = body.result || {};
  const cards = $("summary-cards");
  cards.replaceChildren(
    card("状态", body.status || "-"),
    card("Token", fmtInt(result.total_tokens)),
    card("成本", fmtCost(result.cost)),
    card("耗时", result.duration_ms ? `${(result.duration_ms / 1000).toFixed(1)}s` : "-"),
  );

  $("summary").textContent = result.summary || body.error || "(无结论)";
  const findings = $("findings");
  findings.replaceChildren();
  for (const item of result.key_findings || []) {
    const li = document.createElement("li"); li.textContent = item; findings.append(li);
  }

  const documentWrap = $("document-wrap");
  if (result.document && result.document.markdown) {
    documentWrap.classList.remove("hidden");
    $("document").textContent = result.document.markdown;
  } else {
    documentWrap.classList.add("hidden");
  }

  const reviewWrap = $("review-wrap");
  if (result.review) {
    reviewWrap.classList.remove("hidden");
    $("review").textContent = `verdict = ${result.review.verdict || "-"}（问题 ${result.review.issue_count ?? "-"} 个）`;
    const issues = $("review-issues");
    issues.replaceChildren();
    for (const issue of result.review.issues || []) {
      const li = document.createElement("li");
      li.textContent = `[${issue.severity || "-"}] ${issue.description || ""}`;
      issues.append(li);
    }
  } else {
    reviewWrap.classList.add("hidden");
  }

  const tbody = $("nodes-table").querySelector("tbody");
  tbody.replaceChildren();
  for (const node of result.nodes || []) {
    const tr = document.createElement("tr");
    for (const value of [
      node.node_id, node.agent_role, node.status,
      node.duration_ms != null ? `${(node.duration_ms / 1000).toFixed(1)}s` : "-",
      fmtInt(node.tokens),
    ]) {
      const td = document.createElement("td"); td.textContent = value ?? "-"; tr.append(td);
    }
    tbody.append(tr);
  }

  renderDag(result.dag, result.nodes);

  const actions = $("approval-actions");
  if (body.status === "awaiting_approval") actions.classList.remove("hidden");
  else actions.classList.add("hidden");
}

async function refreshSelected() {
  if (!state.selected) return;
  try {
    const body = await api(`/tasks/${state.selected}`);
    renderDetail(body);
  } catch (error) {
    toast(`加载任务失败：${error.message}`);
  }
}

async function selectTask(taskId) {
  state.selected = taskId;
  await refreshSelected();
  openSSE(taskId);
  loadTasks();
}

async function loadMetrics() {
  try {
    const body = await api("/metrics");
    $("metrics-unit").textContent = `(${body.cost_unit || "credits"})`;
    const cards = $("metric-cards");
    cards.replaceChildren(
      card("任务", fmtInt(body.tasks.total)),
      card("完成", fmtInt(body.tasks.finished)),
      card("Token", fmtInt(body.tokens.total)),
      card("成本", fmtCost(body.cost.total)),
    );
    const agentBody = $("agent-table").querySelector("tbody");
    agentBody.replaceChildren();
    for (const [role, tokens] of Object.entries(body.tokens.by_agent || {})) {
      const tr = document.createElement("tr");
      const a = document.createElement("td"); a.textContent = role;
      const b = document.createElement("td"); b.textContent = fmtInt(tokens);
      tr.append(a, b); agentBody.append(tr);
    }
    const routeBody = $("route-table").querySelector("tbody");
    routeBody.replaceChildren();
    for (const [route, cost] of Object.entries(body.cost.by_route || {})) {
      const tr = document.createElement("tr");
      const a = document.createElement("td"); a.textContent = route;
      const b = document.createElement("td"); b.textContent = fmtCost(cost);
      tr.append(a, b); routeBody.append(tr);
    }
  } catch (error) {
    if (state.key) toast(`加载指标失败：${error.message}`);
  }
}

async function submitTask(event) {
  event.preventDefault();
  const question = $("question").value.trim();
  if (!question) { toast("请输入任务描述"); return; }
  try {
    const body = await api("/tasks", { method: "POST", body: JSON.stringify({ question }) });
    $("submit-hint").textContent = `已提交 ${body.task_id}`;
    $("question").value = "";
    await loadTasks();
    await selectTask(body.task_id);
  } catch (error) {
    toast(`提交失败：${error.message}`);
  }
}

async function decide(decision) {
  if (!state.selected) return;
  try {
    await api(`/tasks/${state.selected}/approval`, {
      method: "POST",
      body: JSON.stringify({ decision, comment: "console" }),
    });
    toast(`已提交审批：${decision}`);
    await refreshSelected();
    await loadTasks();
  } catch (error) {
    toast(`审批失败：${error.message}`);
  }
}

function bind() {
  $("save-key").addEventListener("click", async () => {
    state.key = $("api-key").value.trim();
    sessionStorage.setItem("ma_api_key", state.key);
    toast(state.key ? "Key 已保存（仅在本浏览器 sessionStorage）" : "已清除 Key");
    await checkHealth();
    await loadTasks();
    await loadMetrics();
    if (state.selected) openSSE(state.selected);
  });
  $("refresh").addEventListener("click", () => { loadTasks(); loadMetrics(); });
  $("submit-form").addEventListener("submit", submitTask);
  $("approve").addEventListener("click", () => decide("approved"));
  $("reject").addEventListener("click", () => decide("rejected"));
}

async function init() {
  $("api-key").value = state.key;
  bind();
  await checkHealth();
  if (state.key) {
    await loadTasks();
    await loadMetrics();
  } else {
    $("submit-hint").textContent = "鉴权开启时请先填写 API Key（operator/admin 才能提交）。";
  }
  state.timer = setInterval(() => {
    if (document.hidden) return;
    loadTasks();
    loadMetrics();
    if (state.selected) refreshSelected();
  }, 10000);
}

init();
