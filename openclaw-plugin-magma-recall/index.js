import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const TAG = "[magma-recall]";
const STRIP_RE = /<magma-memories>[\s\S]*?<\/magma-memories>\s*/g;
const PROMPT_CACHE_TTL_MS = 10 * 60 * 1000;
const PROMPT_CACHE_MAX_SIZE = 1000;
const OPS_ANCHOR_RULES = [
  { id: "ops:magma:model-runtime-2026-05-26", terms: ["MiniLM", "384", "bge-small", "bge-small-zh", "embedding", "DeepSeek", "OpenRouter", "LLM"] },
  { id: "ops:magma:health-signals", terms: ["recent_capture", "doctor", "yellow", "红黄绿"] },
  { id: "ops:magma:mcp-proxy-8902", terms: ["8902", "mcp", "mcp_proxy", "薄代理", "http_proxy"] },
  { id: "ops:openclaw:version-pin-2026-5-20", terms: ["2026.5.20", "5.20", "5.22", "版本 pin", "pin"] },
  { id: "ops:magma:p0-ops-suite", terms: ["p0", "magma_ops.py", "magma_doctor.py", "runbook", "三件套"] },
  { id: "ops:magma:yunying-source-agent", terms: ["yunying", "source_agent_id", "source_agents", "subagent"] },
];

let apiStarted = false;
const pendingPrompts = new Map();
const pendingRecalls = new Map();

function resolveConfig(raw = {}) {
  return {
    enabled: raw.enabled !== false,
    apiBaseUrl: String(raw.apiBaseUrl || "http://127.0.0.1:8901").replace(/\/+$/, ""),
    topK: Number.isFinite(raw.topK) ? Math.max(1, Math.min(10, Math.floor(raw.topK))) : 6,
    timeoutMs: Number.isFinite(raw.timeoutMs) ? Math.max(200, Math.floor(raw.timeoutMs)) : 12000,
    scoreThreshold: Number.isFinite(raw.scoreThreshold) ? raw.scoreThreshold : 0.35,
    captureEnabled: raw.capture?.enabled !== false,
    captureTtlDays: Number.isFinite(raw.capture?.ttlDays) ? Math.max(1, Math.floor(raw.capture.ttlDays)) : 180,
    captureMaxChars: Number.isFinite(raw.capture?.maxChars) ? Math.max(200, Math.floor(raw.capture.maxChars)) : 4000,
    autoStartApi: raw.autoStartApi !== false,
    python: String(raw.python || "python"),
    magmaCwd: String(raw.magmaCwd || "C:\\openclaw-magma"),
    excludeAgents: Array.isArray(raw.excludeAgents) ? raw.excludeAgents.map(String) : [],
  };
}

function shouldSkipAgent(agentId, excludeAgents) {
  if (!agentId) return false;
  return excludeAgents.some((pattern) => {
    if (pattern === agentId) return true;
    if (!pattern.includes("*")) return false;
    const re = new RegExp(`^${pattern.split("*").map(escapeRegex).join(".*")}$`);
    return re.test(agentId);
  });
}

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function parseApiPort(apiBaseUrl) {
  try {
    const url = new URL(apiBaseUrl);
    return url.port || (url.protocol === "https:" ? "443" : "80");
  } catch {
    return "8901";
  }
}

function sweepPromptCache() {
  const now = Date.now();
  for (const [key, entry] of pendingPrompts) {
    if (now - entry.ts > PROMPT_CACHE_TTL_MS) pendingPrompts.delete(key);
  }
  if (pendingPrompts.size <= PROMPT_CACHE_MAX_SIZE) return;
  const entries = [...pendingPrompts.entries()].sort((a, b) => a[1].ts - b[1].ts);
  for (const [key] of entries.slice(0, entries.length - PROMPT_CACHE_MAX_SIZE)) {
    pendingPrompts.delete(key);
  }
}

async function fetchWithTimeout(url, options, timeoutMs) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

async function isApiHealthy(cfg) {
  try {
    const res = await fetchWithTimeout(`${cfg.apiBaseUrl}/api/v1/health`, {}, cfg.timeoutMs);
    return res.ok;
  } catch {
    return false;
  }
}

async function ensureApiStarted(cfg, logger) {
  if (!cfg.autoStartApi || apiStarted) return;
  if (await isApiHealthy(cfg)) {
    apiStarted = true;
    return;
  }
  if (!fs.existsSync(cfg.magmaCwd)) {
    logger.warn?.(`${TAG} MAGMA cwd does not exist: ${cfg.magmaCwd}`);
    return;
  }
  const child = spawn(cfg.python, ["-m", "magma.api.server"], {
    cwd: cfg.magmaCwd,
    detached: true,
    stdio: "ignore",
    windowsHide: true,
    env: {
      ...process.env,
      HF_ENDPOINT: process.env.HF_ENDPOINT || "https://huggingface.co",
      MAGMA_API_PORT: parseApiPort(cfg.apiBaseUrl),
    },
  });
  child.unref();
  apiStarted = true;
  logger.info?.(`${TAG} started MAGMA API via ${cfg.python} -m magma.api.server`);
}

function extractUserText(event) {
  if (typeof event?.prompt === "string" && event.prompt.trim()) return event.prompt.trim();
  const messages = Array.isArray(event?.messages) ? event.messages : [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg?.role !== "user") continue;
    const content = msg.content;
    if (typeof content === "string") return content.trim();
    if (Array.isArray(content)) {
      return content
        .filter((part) => part?.type === "text" && typeof part.text === "string")
        .map((part) => part.text)
        .join("\n")
        .trim();
    }
  }
  return "";
}

async function queryMagma(cfg, text, context = {}) {
  const filters = { include_related: true, related_limit: 2, include_versions: true, version_limit: 2, pool_size: 5000 };
  if (context.agentId) filters.current_agent_id = context.agentId;
  const res = await fetchWithTimeout(
    `${cfg.apiBaseUrl}/api/v1/query`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, top_k: cfg.topK, filters }),
    },
    cfg.timeoutMs,
  );
  if (!res.ok) throw new Error(`MAGMA query failed: HTTP ${res.status}`);
  const body = await res.json();
  const results = Array.isArray(body.results) ? body.results : [];
  return await mergeOperationalAnchors(cfg, text, results);
}

function anchorIdsForText(text) {
  const lower = String(text || "").toLowerCase();
  return OPS_ANCHOR_RULES
    .filter((rule) => rule.terms.some((term) => lower.includes(term.toLowerCase())))
    .map((rule) => rule.id);
}

async function fetchMagmaNode(cfg, id) {
  const res = await fetchWithTimeout(
    `${cfg.apiBaseUrl}/api/v1/nodes/${encodeURIComponent(id)}`,
    {},
    cfg.timeoutMs,
  );
  if (!res.ok) return null;
  const body = await res.json();
  return body?.node || null;
}

async function mergeOperationalAnchors(cfg, text, results) {
  const merged = [...results];
  const seen = new Set(merged.map((item) => item?.id).filter(Boolean));
  for (const id of anchorIdsForText(text)) {
    if (seen.has(id)) continue;
    const node = await fetchMagmaNode(cfg, id).catch(() => null);
    if (!node) continue;
    merged.unshift({
      ...node,
      score: 1.25,
      semantic_score: 0,
      keyword_score: 1.25,
      retrieval_source: "operational_anchor",
      memory_scope: node.properties?.memory_scope || "system",
      provenance: {
        agent_id: node.source_agent_id || node.properties?.source_agent_id,
        source: node.properties?.source,
        layer: node.properties?.layer,
      },
      source_agent_id: node.source_agent_id || node.properties?.source_agent_id,
      memory_source: node.properties?.source,
    });
    seen.add(id);
  }
  return merged.slice(0, cfg.topK);
}

async function captureMagma(cfg, payload) {
  const res = await fetchWithTimeout(
    `${cfg.apiBaseUrl}/api/v1/capture`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    Math.max(cfg.timeoutMs, 3000),
  );
  if (!res.ok) throw new Error(`MAGMA capture failed: HTTP ${res.status}`);
  return await res.json();
}

async function feedbackMagma(cfg, payload) {
  const res = await fetchWithTimeout(
    `${cfg.apiBaseUrl}/api/v1/feedback`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
    Math.max(cfg.timeoutMs, 3000),
  );
  if (!res.ok) throw new Error(`MAGMA feedback failed: HTTP ${res.status}`);
  return await res.json();
}

function previewText(result) {
  const props = result.properties || {};
  const raw =
    props.title ||
    props.name ||
    props.content ||
    props.message ||
    JSON.stringify(props);
  return String(raw || "").replace(/\s+/g, " ").slice(0, 240);
}

function resultSearchText(result) {
  const props = result.properties || {};
  const entities = Array.isArray(result.query_entities) ? result.query_entities : [];
  const nodeEntities = Array.isArray(props.entities) ? props.entities : [];
  return [
    result.id,
    result.label,
    props.title,
    props.name,
    props.content,
    props.message,
    props.source_file,
    props.version_key,
    ...entities.map((item) => item?.name),
    ...nodeEntities.map((item) => item?.name),
  ].filter(Boolean).join("\n");
}

function termsForResult(result) {
  const text = resultSearchText(result);
  const terms = new Set();
  for (const raw of text.split(/[\s,，。；;:：|/\\()[\]{}"'`<>]+/)) {
    const term = raw.trim();
    if (term.length >= 3 && term.length <= 40) terms.add(term);
  }
  const props = result.properties || {};
  for (const entity of props.entities || []) {
    if (entity?.name) terms.add(String(entity.name));
  }
  if (props.name) terms.add(String(props.name));
  if (props.title) terms.add(String(props.title).slice(0, 40));
  return [...terms].slice(0, 20);
}

function findUsedMemories(results, assistantText) {
  const text = String(assistantText || "");
  if (!text.trim()) return [];
  return results.filter((result) => {
    const terms = termsForResult(result);
    return terms.some((term) => text.includes(term));
  });
}

function recallEventId(ctx, started) {
  const key = cacheKey(ctx) || ctx?.agentId || "unknown";
  return `recall:${key}:${started}`;
}

function compactText(result, maxChars = 160) {
  return previewText(result).slice(0, maxChars);
}

function memoryTitle(result) {
  const props = result.properties || {};
  return props.title || props.name || result.label || "memory";
}

function provenanceText(result) {
  const provenance = result.provenance || {};
  const parts = [];
  const source = provenance.source || result.memory_source;
  const agent = provenance.agent_id || result.source_agent_id;
  const role = provenance.role;
  const layer = provenance.layer;
  const scope = result.memory_scope;
  if (source) parts.push(`source=${source}`);
  if (agent) parts.push(`agent=${agent}`);
  if (role) parts.push(`role=${role}`);
  if (layer) parts.push(`layer=${layer}`);
  if (scope) parts.push(`scope=${scope}`);
  return parts.length ? ` ${parts.join(" ")}` : "";
}

function relationPreview(edge) {
  const neighbor = edge?.neighbor || {};
  const props = neighbor.properties || {};
  const raw = [props.title || props.name, props.content || props.message].filter(Boolean).join(" - ") || neighbor.id || "";
  const text = String(raw || "").replace(/\s+/g, " ").slice(0, 120);
  return `${edge.relation || "related_to"} -> [${neighbor.id || "unknown"}] ${neighbor.label || "memory"}: ${text}`;
}

function sourceLabel(result) {
  const provenance = result.provenance || {};
  const parts = [];
  if (result.memory_scope) parts.push(`scope=${result.memory_scope}`);
  if (provenance.agent_id || result.source_agent_id) parts.push(`agent=${provenance.agent_id || result.source_agent_id}`);
  if (provenance.source || result.memory_source) parts.push(`source=${provenance.source || result.memory_source}`);
  if (provenance.role) parts.push(`role=${provenance.role}`);
  return parts.join(", ") || "scope=unknown";
}

function groupByScope(results) {
  const groups = new Map();
  for (const item of results) {
    const scope = item.memory_scope || "general";
    if (!groups.has(scope)) groups.set(scope, []);
    groups.get(scope).push(item);
  }
  return groups;
}

function buildFactBlocks(results) {
  const groups = groupByScope(results);
  const lines = ["Compressed facts:"];
  for (const scope of ["product", "system", "mixed", "general"]) {
    const items = groups.get(scope) || [];
    if (items.length === 0) continue;
    const facts = items.slice(0, 2).map((item) => {
      const score = typeof item.score === "number" ? item.score.toFixed(3) : "n/a";
      return `[${item.id}] ${compactText(item, 140)} (score=${score})`;
    });
    lines.push(`- ${scope}: ${facts.join(" | ")}`);
  }
  return lines;
}

function buildRelationBlocks(results) {
  const lines = [];
  const seen = new Set();
  for (const item of results) {
    const related = Array.isArray(item.related_context) ? item.related_context.slice(0, 2) : [];
    for (const edge of related) {
      const key = `${edge.source}:${edge.relation}:${edge.target}`;
      if (seen.has(key)) continue;
      seen.add(key);
      lines.push(`- ${item.id}: ${relationPreview(edge)}`);
      if (lines.length >= 4) return ["Useful relations:", ...lines];
    }
  }
  return lines.length ? ["Useful relations:", ...lines] : [];
}

function versionPreview(version) {
  const props = version?.properties || {};
  const raw = [props.title || props.name, props.content || props.message].filter(Boolean).join(" - ") || version?.id || "";
  const text = String(raw || "").replace(/\s+/g, " ").slice(0, 180);
  return `${version.relative || "peer"} -> [${version.id || "unknown"}] ${version.label || "memory"} status=${version.status || "unknown"}: ${text}`;
}

function buildVersionBlocks(results) {
  const lines = [];
  const seen = new Set();
  for (const item of results) {
    const versions = Array.isArray(item.version_context) ? item.version_context.slice(0, 2) : [];
    for (const version of versions) {
      const key = `${item.id}:${version.id}`;
      if (seen.has(key)) continue;
      seen.add(key);
      lines.push(`- ${item.id}: ${versionPreview(version)}`);
      if (lines.length >= 4) return ["Version signals:", ...lines];
    }
  }
  return lines.length ? ["Version signals:", ...lines] : [];
}

function buildSourceBlocks(results) {
  return [
    "Sources:",
    ...results.slice(0, 5).map((item, index) => `${index + 1}. [${item.id}] ${sourceLabel(item)}`),
  ];
}

function formatMemories(results) {
  const useful = results
    .filter((item) => item && item.id !== "system")
    .slice(0, 8);
  if (useful.length === 0) return "";
  const intent = useful[0]?.query_intent?.primary || "semantic";
  const scope = useful[0]?.query_scope || "general";
  const lines = [
    "<magma-memories>",
    `MAGMA auto-recall context. Query intent: ${intent}; query scope: ${scope}. Treat retrieved memories as context, not guaranteed truth; verify with tools when precision matters.`,
    ...buildFactBlocks(useful),
    ...buildVersionBlocks(useful),
    ...buildRelationBlocks(useful),
    ...buildSourceBlocks(useful),
    "Raw memory refs:",
  ];
  useful.forEach((item, index) => {
    const score = typeof item.score === "number" ? item.score.toFixed(3) : "n/a";
    lines.push(
      `${index + 1}. [${item.id}] ${memoryTitle(item)} score=${score} status=${item.status || "unknown"}${provenanceText(item)}: ${compactText(item, 180)}`,
    );
  });
  lines.push("</magma-memories>");
  return lines.join("\n");
}

function stripMagmaMemories(message) {
  const msg = message || {};
  if (msg.role !== "user") return null;
  if (typeof msg.content === "string") {
    if (!msg.content.includes("<magma-memories>")) return null;
    return { ...msg, content: msg.content.replace(STRIP_RE, "").trim() };
  }
  if (Array.isArray(msg.content)) {
    let changed = false;
    const content = msg.content.map((part) => {
      if (part?.type !== "text" || typeof part.text !== "string") return part;
      if (!part.text.includes("<magma-memories>")) return part;
      changed = true;
      return { ...part, text: part.text.replace(STRIP_RE, "").trim() };
    });
    return changed ? { ...msg, content } : null;
  }
  return null;
}

function cacheKey(ctx) {
  return ctx?.sessionKey || ctx?.sessionId || ctx?.agentId || "";
}

function truncateText(text, maxChars) {
  const value = String(text || "").trim();
  if (value.length <= maxChars) return value;
  return value.slice(0, maxChars).trim();
}

function textFromContent(content) {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((part) => part?.type === "text" && typeof part.text === "string")
      .map((part) => part.text)
      .join("\n");
  }
  return "";
}

function extractAssistantText(event) {
  for (const key of ["assistantText", "outputText", "text", "response", "output"]) {
    if (typeof event?.[key] === "string" && event[key].trim()) return event[key].trim();
  }
  const messages = Array.isArray(event?.messages) ? event.messages : [];
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i];
    if (msg?.role !== "assistant") continue;
    const text = textFromContent(msg.content).trim();
    if (text) return text;
  }
  return "";
}

function writeAudit(cfg, payload) {
  try {
    const logDir = path.join(process.env.USERPROFILE || process.env.HOME || cfg.magmaCwd, ".openclaw", "logs");
    fs.mkdirSync(logDir, { recursive: true });
    fs.appendFileSync(
      path.join(logDir, "magma-recall.jsonl"),
      `${JSON.stringify({ ts: new Date().toISOString(), ...payload })}\n`,
      "utf8",
    );
  } catch {
    // Audit logging should never block the agent path.
  }
}

export default function register(api) {
  const cfg = resolveConfig(api.pluginConfig);
  if (!cfg.enabled) {
    api.logger.info?.(`${TAG} disabled`);
    return;
  }

  api.logger.info?.(`${TAG} registering auto recall topK=${cfg.topK}, timeout=${cfg.timeoutMs}ms`);
  void ensureApiStarted(cfg, api.logger);

  api.on("before_prompt_build", async (event, ctx) => {
    const started = Date.now();
    const agentId = ctx?.agentId || "";
    if (shouldSkipAgent(agentId, cfg.excludeAgents)) return;

    const userText = extractUserText(event);
    if (!userText) return;
    const key = cacheKey(ctx);
    if (key) {
      pendingPrompts.set(key, { text: userText, ts: Date.now() });
      sweepPromptCache();
    }

    try {
      await ensureApiStarted(cfg, api.logger);
      const results = await queryMagma(cfg, userText, { agentId });
      const filtered = results.filter((item) => typeof item.score !== "number" || item.score >= cfg.scoreThreshold);
      const prependContext = formatMemories(filtered);
      const durationMs = Date.now() - started;
      const eventId = recallEventId(ctx, started);
      const keyForRecall = cacheKey(ctx);
      if (keyForRecall && filtered.length > 0) {
        pendingRecalls.set(keyForRecall, {
          eventId,
          query: userText,
          results: filtered.slice(0, cfg.topK),
          agentId,
          sessionKey: ctx?.sessionKey,
          ts: Date.now(),
        });
      }
      writeAudit(cfg, {
        agentId,
        sessionKey: ctx?.sessionKey,
        eventId,
        durationMs,
        resultCount: filtered.length,
        recalled: filtered.map((item) => item.id),
        queryPreview: userText.slice(0, 120),
      });
      if (!prependContext) {
        api.logger.info?.(`${TAG} recall complete (${durationMs}ms), no context`);
        return;
      }
      api.logger.info?.(`${TAG} recall complete (${durationMs}ms), injected ${filtered.length} memories`);
      return { prependContext };
    } catch (err) {
      const durationMs = Date.now() - started;
      api.logger.warn?.(`${TAG} recall skipped after ${durationMs}ms: ${err instanceof Error ? err.message : String(err)}`);
      writeAudit(cfg, {
        agentId,
        sessionKey: ctx?.sessionKey,
        durationMs,
        error: err instanceof Error ? err.message : String(err),
        queryPreview: userText.slice(0, 120),
      });
    }
  });

  api.on("before_message_write", (event) => {
    const cleaned = stripMagmaMemories(event?.message);
    if (!cleaned) return;
    return { message: cleaned };
  });

  if (cfg.captureEnabled) {
    api.logger.info?.(`${TAG} registering auto capture ttl=${cfg.captureTtlDays}d`);
    api.on("agent_end", async (event, ctx) => {
      const started = Date.now();
      const agentId = ctx?.agentId || "";
      if (shouldSkipAgent(agentId, cfg.excludeAgents)) return;
      if (event && event.success === false) return;

      const key = cacheKey(ctx);
      const cached = key ? pendingPrompts.get(key) : undefined;
      const recallBatch = key ? pendingRecalls.get(key) : undefined;
      if (key) pendingPrompts.delete(key);
      if (key) pendingRecalls.delete(key);

      const userText = truncateText(cached?.text || extractUserText(event), cfg.captureMaxChars);
      const assistantText = truncateText(extractAssistantText(event), cfg.captureMaxChars);
      if (!userText && !assistantText) return;

      try {
        await ensureApiStarted(cfg, api.logger);
        let feedbackResult = null;
        if (recallBatch && assistantText) {
          const used = findUsedMemories(recallBatch.results, assistantText);
          feedbackResult = await feedbackMagma(cfg, {
            event_id: recallBatch.eventId,
            query: recallBatch.query,
            agent_id: recallBatch.agentId || undefined,
            session_key: recallBatch.sessionKey,
            recalled: recallBatch.results.map((item) => ({
              id: item.id,
              score: item.score,
              memory_scope: item.memory_scope,
            })),
            used: used.map((item) => ({ id: item.id })),
          });
        }
        const result = await captureMagma(cfg, {
          user_text: userText,
          assistant_text: assistantText,
          agent_id: agentId || undefined,
          session_key: ctx?.sessionKey,
          session_id: ctx?.sessionId,
          source: "openclaw_auto_capture",
          ttl_days: cfg.captureTtlDays,
        });
        const durationMs = Date.now() - started;
        writeAudit(cfg, {
          type: "capture",
          agentId,
          sessionKey: ctx?.sessionKey,
          durationMs,
          written: result.written || [],
          count: result.count || 0,
          feedback: feedbackResult?.feedback,
          queryPreview: userText.slice(0, 120),
        });
        const usedCount = feedbackResult?.feedback?.used || 0;
        api.logger.info?.(`${TAG} capture complete (${durationMs}ms), wrote ${result.count || 0} nodes, feedback used ${usedCount}`);
      } catch (err) {
        const durationMs = Date.now() - started;
        api.logger.warn?.(`${TAG} capture skipped after ${durationMs}ms: ${err instanceof Error ? err.message : String(err)}`);
        writeAudit(cfg, {
          type: "capture",
          agentId,
          sessionKey: ctx?.sessionKey,
          durationMs,
          error: err instanceof Error ? err.message : String(err),
          queryPreview: userText.slice(0, 120),
        });
      }
    });
  }
}
