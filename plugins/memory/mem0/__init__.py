"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search with reranking, and
automatic deduplication via the Mem0 Platform API.

Original PR #2933 by kartik-mem0, adapted to MemoryProvider ABC.

Config via environment variables:
  MEM0_API_KEY       — Mem0 Platform API key (required)
  MEM0_USER_ID       — User identifier (default: hermes-user)
  MEM0_AGENT_ID      — Agent identifier (default: hermes)

Or via $HERMES_HOME/mem0.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after this many consecutive failures, pause API calls
# for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD = 5
_BREAKER_COOLDOWN_SECS = 120


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    """Load config from env vars, with $HERMES_HOME/mem0.json overrides.

    Environment variables provide defaults; mem0.json (if present) overrides
    individual keys.  This avoids a silent failure when the JSON file exists
    but is missing fields like ``api_key`` that the user set in ``.env``.
    """
    from hermes_constants import get_hermes_home

    config = {
        "api_key": os.environ.get("MEM0_API_KEY", ""),
        "user_id": os.environ.get("MEM0_USER_ID", "hermes-user"),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "rerank": True,
        "keyword_search": False,
    }

    config_path = get_hermes_home() / "mem0.json"
    if config_path.exists():
        try:
            file_cfg = json.loads(config_path.read_text(encoding="utf-8"))
            config.update({k: v for k, v in file_cfg.items()
                           if v is not None and v != ""})
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

PROFILE_SCHEMA = {
    "name": "mem0_profile",
    "description": (
        "Retrieve all stored memories about the user — preferences, facts, "
        "project context. Fast, no reranking. Use at conversation start."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "mem0_search",
    "description": (
        "Search memories by meaning. Returns relevant facts ranked by similarity. "
        "Set rerank=true for higher accuracy on important queries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "rerank": {"type": "boolean", "description": "Enable reranking for precision (default: false)."},
            "top_k": {"type": "integer", "description": "Max results (default: 10, max: 50)."},
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "mem0_conclude",
    "description": (
        "Store a durable fact about the user. Stored verbatim (no LLM extraction). "
        "Use for explicit preferences, corrections, or decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {"type": "string", "description": "The fact to store."},
        },
        "required": ["conclusion"],
    },
}


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 Platform memory with server-side extraction and semantic search."""

    def __init__(self):
        self._config = None
        self._client = None
        self._client_lock = threading.Lock()
        self._api_key = ""
        self._user_id = "hermes-user"
        self._agent_id = "hermes"
        self._rerank = True
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._sync_thread = None
        # Circuit breaker state
        self._consecutive_failures = 0
        self._breaker_open_until = 0.0

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        # Support both platform mode (api_key) and local mode
        return bool(cfg.get("api_key")) or cfg.get("mode") == "local"

    def save_config(self, values, hermes_home):
        """Write config to $HERMES_HOME/mem0.json."""
        import json
        from pathlib import Path
        config_path = Path(hermes_home) / "mem0.json"
        existing = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
            except Exception:
                pass
        existing.update(values)
        from utils import atomic_json_write
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": True, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "true", "choices": ["true", "false"]},
        ]

    def _get_client(self):
        """Thread-safe client accessor with lazy initialization."""
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                cfg = _load_config()
                if cfg.get("mode") == "local":
                    # Local mode: use Memory.from_config()
                    # ⚡ Force offline mode — prevent transformers/hf_hub from downloading
                    # config files every time the model loads. Model files are already local.
                    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                    from mem0 import Memory
                    self._client = Memory.from_config({
                        'llm': {
                            'provider': 'openai',
                            'config': {
                                'api_key': 'local',
                                'openai_base_url': cfg.get("llm_base_url", "http://localhost:1234/v1"),
                                'model': cfg.get("llm_model", "qwen3")
                            }
                        },
                        'embedder': {
                            'provider': 'huggingface',
                            'config': {
                                'model': cfg.get("embedder_model", "/home/herocco/bge/bge-large-zh-v1.5")
                            }
                        },
                        'vector_store': {
                            'provider': 'qdrant',
                            'config': {
                                'collection_name': 'mem0',
                                'embedding_model_dims': cfg.get("embedding_dims", 1024),
                                'host': cfg.get("qdrant_host", "localhost"),
                                'port': cfg.get("qdrant_port", 6333)
                            }
                        },
                    })
                else:
                    # Platform mode: use MemoryClient with API key
                    from mem0 import MemoryClient
                    self._client = MemoryClient(api_key=self._api_key)
                return self._client
            except ImportError:
                raise RuntimeError("mem0 package not installed. Run: pip install mem0ai")

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        if self._consecutive_failures < _BREAKER_THRESHOLD:
            return False
        if time.monotonic() >= self._breaker_open_until:
            # Cooldown expired — reset and allow a retry
            self._consecutive_failures = 0
            return False
        return True

    def _record_success(self):
        self._consecutive_failures = 0

    def _record_failure(self):
        self._consecutive_failures += 1
        if self._consecutive_failures >= _BREAKER_THRESHOLD:
            self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.",
                self._consecutive_failures, _BREAKER_COOLDOWN_SECS,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._api_key = self._config.get("api_key", "")
        # Prefer gateway-provided user_id for per-user memory scoping;
        # fall back to config/env default for CLI (single-user) sessions.
        self._user_id = kwargs.get("user_id") or self._config.get("user_id", "hermes-user")
        self._agent_id = self._config.get("agent_id", "hermes")
        self._rerank = self._config.get("rerank", True)

    def _read_filters(self) -> Dict[str, Any]:
        """Filters for search/get_all — scoped to user only for cross-session recall."""
        return {"user_id": self._user_id}

    def _write_filters(self) -> Dict[str, Any]:
        """Filters for add — scoped to user + agent for attribution."""
        return {"user_id": self._user_id, "agent_id": self._agent_id}

    @staticmethod
    def _unwrap_results(response: Any) -> list:
        """Normalize Mem0 API response — v2 wraps results in {"results": [...]}."""
        if isinstance(response, dict):
            return response.get("results", [])
        if isinstance(response, list):
            return response
        return []

    def system_prompt_block(self) -> str:
        return (
            "# Mem0 Memory\n"
            f"Active. User: {self._user_id}.\n"
            "Use mem0_search to find memories, mem0_conclude to store facts, "
            "mem0_profile for a full overview.\n"
            "\n"
            "## Memory Format\n"
            "Each memory has a prefix: `[Updated N days ago | YYYY-MM-DD | 高频/中频/低频]`\n"
            "- 高频 (>20次访问): 重要事实，优先信任\n"
            "- 中频 (5-20次): 常规信息\n"
            "- 低频 (<5次): 可能已过时\n"
            "Conflicts are marked: `⚠️ 可能与第N条冲突(相似度X.XX)`\n"
            "\n"
            "## Conflict Resolution Rules (when conflicting memories appear)\n"
            "1. **时效优先**: 选择更新时间较近的记忆（Updated N days ago 较小）\n"
            "2. **频次优先** (时间差 < 3天): 选择访问频次较高的记忆（高频 > 中频 > 低频）\n"
            "3. **冲突确认** (权重相当时): 不要自行决定丢弃任何一方，在回复中委婉提及两种可能性"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Mem0 Memory\n{result}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if self._is_breaker_open():
            return

        def _run():
            try:
                import subprocess
                import re
                
                # Over-fetching: retrieve top_k=20, deduplicate, then inject top 5
                # This solves high-frequency memory occupation and improves recall
                top_k_fetch = 20
                top_k_inject = 5
                
                MEM0_SERVER = "/media/data/mem0/mem0_server.py"
                MEM0_PYTHON = "/media/data/mem0/.venv/bin/python"

                def run_with_retry(cmd, timeout=30, max_retries=3):
                    """Run subprocess with exponential backoff retry."""
                    last_error = None
                    for attempt in range(max_retries):
                        try:
                            result = subprocess.run(
                                cmd, capture_output=True, text=True, timeout=timeout
                            )
                            if result.returncode == 0:
                                return result
                            last_error = f"Command failed (attempt {attempt + 1}): {result.stderr}"
                        except subprocess.TimeoutExpired as e:
                            last_error = f"Timeout (attempt {attempt + 1}): {e}"
                        except Exception as e:
                            last_error = f"Error (attempt {attempt + 1}): {e}"

                        # Exponential backoff: 1s, 2s, 4s
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)

                    raise RuntimeError(last_error)

                # Step 1: Search with --no-track (decouple search from tracking)
                result = run_with_retry(
                    [
                        MEM0_PYTHON, MEM0_SERVER,
                        "search", query, self._user_id, str(top_k_fetch),
                        "true" if self._rerank else "false", "--no-track"
                    ],
                    timeout=30
                )
                
                data = json.loads(result.stdout)
                results = self._unwrap_results(data)
                
                if not results:
                    return
                
                # Step 2: Dual-threshold deduplication (keep Qdrant original order)
                # - cosine > 0.95 + exact text match: discard (pure redundant), add to shadow_ids
                # - cosine > 0.95 + text differs: downgrade to conflict retention
                # - 0.85 < cosine <= 0.95: retain + mark conflict
                # - cosine <= 0.85: normal injection
                
                # Use text-based similarity (faster than embeddings, no extra deps)
                from difflib import SequenceMatcher
                
                def text_similarity(text1, text2):
                    """Compute text similarity using SequenceMatcher (0-1 scale)."""
                    if not text1 or not text2:
                        return 0.0
                    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
                
                def is_exact_duplicate(text1, text2):
                    """Check if texts are identical after removing punctuation and spaces."""
                    clean1 = re.sub(r'[^\w\s]', '', text1).strip()
                    clean2 = re.sub(r'[^\w\s]', '', text2).strip()
                    return clean1 == clean2
                
                kept = []           # Memories to inject
                shadow_ids = []     # Pure redundant memories (for touch only)
                conflicts = []      # Conflict pairs: (mem, conflict_with_mem, similarity)
                
                # Keep Qdrant original order (similarity from high to low)
                for mem in results:
                    mem_id = mem.get('id', '')
                    mem_text = mem.get('memory', '')
                    
                    is_duplicate = False
                    conflict_with = None
                    max_sim = 0.0
                    
                    for kept_mem in kept:
                        cos_sim = text_similarity(mem_text, kept_mem.get('memory', ''))
                        
                        if cos_sim > 0.95:
                            if is_exact_duplicate(mem_text, kept_mem.get('memory', '')):
                                shadow_ids.append(mem_id)
                                is_duplicate = True
                                break
                            else:
                                # Text differs, downgrade to conflict retention
                                if cos_sim > max_sim:
                                    conflict_with = kept_mem
                                    max_sim = cos_sim
                        elif cos_sim > 0.85 and cos_sim > max_sim:
                            conflict_with = kept_mem
                            max_sim = cos_sim
                    
                    if not is_duplicate:
                        kept.append(mem)
                        if conflict_with is not None:
                            conflicts.append((mem, conflict_with, max_sim))
                
                # Step 3: Track inject_ids and touch shadow_ids
                if kept or shadow_ids:
                    # Track inject_ids (full track: update last_accessed_at + increment access_count)
                    inject_ids = [m.get('id', '') for m in kept]
                    if inject_ids:
                        run_with_retry(
                            [
                                MEM0_PYTHON, MEM0_SERVER,
                                "track", json.dumps(inject_ids)
                            ],
                            timeout=10
                        )
                    
                    # Touch shadow_ids (touch only: update last_accessed_at, no access_count change)
                    if shadow_ids:
                        run_with_retry(
                            [
                                MEM0_PYTHON, MEM0_SERVER,
                                "touch", json.dumps(shadow_ids)
                            ],
                            timeout=10
                        )
                
                # Step 4: Format memories with frequency labels and conflict markers
                from datetime import datetime, timezone
                
                def get_frequency_label(access_count):
                    if access_count > 20:
                        return "高频"
                    elif access_count >= 5:
                        return "中频"
                    else:
                        return "低频"
                
                def format_memory(mem, conflict_info=None):
                    now = datetime.now(timezone.utc)
                    updated_str = mem.get('updated_at', '')
                    days_ago = 0
                    if updated_str:
                        try:
                            updated = datetime.fromisoformat(updated_str.replace('Z', '+00:00'))
                            if updated.tzinfo is None:
                                updated = updated.replace(tzinfo=timezone.utc)
                            days_ago = max(0, (now - updated).days)
                        except Exception:
                            pass
                    
                    # Access count is in metadata field
                    metadata = mem.get('metadata', {}) or {}
                    access_count = metadata.get('access_count', 0)
                    freq = get_frequency_label(access_count)
                    
                    prefix = f"[Updated {days_ago} days ago | {datetime.now(timezone.utc).strftime('%Y-%m-%d')} | {freq}]"
                    
                    if conflict_info:
                        target_index, sim = conflict_info
                        note = f" ⚠️ 可能与第{target_index}条冲突(相似度{sim:.2f})"
                        return f"- {prefix}{note}：{mem.get('memory', '')}"
                    else:
                        return f"- {prefix} {mem.get('memory', '')}"
                
                # Build final output (top 5 after deduplication)
                final_kept = kept[:top_k_inject]
                lines = []
                for i, mem in enumerate(final_kept):
                    # Check if this memory has a conflict
                    conflict_info = None
                    for c_mem, c_with, c_sim in conflicts:
                        if c_mem.get('id') == mem.get('id'):
                            # Find the index of the conflict target
                            for j, k_mem in enumerate(final_kept):
                                if k_mem.get('id') == c_with.get('id'):
                                    conflict_info = (j + 1, c_sim)
                                    break
                    
                    lines.append(format_memory(mem, conflict_info))
                
                with self._prefetch_lock:
                    self._prefetch_result = "\n".join(lines)
                
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)

        self._prefetch_thread = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        self._prefetch_thread.start()

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._is_breaker_open():
            return

        def _sync():
            try:
                client = self._get_client()
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content},
                ]
                client.add(messages, **self._write_filters())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        # Wait for any previous sync before starting a new one
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)

        self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
        self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [PROFILE_SCHEMA, SEARCH_SCHEMA, CONCLUDE_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._is_breaker_open():
            return json.dumps({
                "error": "Mem0 API temporarily unavailable (multiple consecutive failures). Will retry automatically."
            })

        try:
            client = self._get_client()
        except Exception as e:
            return tool_error(str(e))

        if tool_name == "mem0_profile":
            try:
                memories = self._unwrap_results(client.get_all(filters=self._read_filters()))
                self._record_success()
                if not memories:
                    return json.dumps({"result": "No memories stored yet."})
                lines = [m.get("memory", "") for m in memories if m.get("memory")]
                return json.dumps({"result": "\n".join(lines), "count": len(lines)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to fetch profile: {e}")

        elif tool_name == "mem0_search":
            query = args.get("query", "")
            if not query:
                return tool_error("Missing required parameter: query")
            rerank = args.get("rerank", False)
            top_k = min(int(args.get("top_k", 10)), 50)
            try:
                results = self._unwrap_results(client.search(
                    query=query,
                    filters=self._read_filters(),
                    rerank=rerank,
                    top_k=top_k,
                ))
                self._record_success()
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                items = [{"memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
                return json.dumps({"results": items, "count": len(items)})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Search failed: {e}")

        elif tool_name == "mem0_conclude":
            conclusion = args.get("conclusion", "")
            if not conclusion:
                return tool_error("Missing required parameter: conclusion")
            try:
                client.add(
                    [{"role": "user", "content": conclusion}],
                    **self._write_filters(),
                    infer=False,
                )
                self._record_success()
                return json.dumps({"result": "Fact stored."})
            except Exception as e:
                self._record_failure()
                return tool_error(f"Failed to store: {e}")

        return tool_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        with self._client_lock:
            self._client = None


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
