# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, automatic deduplication, and conflict resolution.

## Modes

### Platform Mode (Cloud)

Uses Mem0's hosted API for memory management.

**Requirements**:
- `pip install mem0ai`
- Mem0 API key from [app.mem0.ai](https://app.mem0.ai)

**Setup**:
```bash
hermes memory setup    # select "mem0"
```

Or manually:
```bash
hermes config set memory.provider mem0
echo "MEM0_API_KEY=*** >> ~/.hermes/.env
```

### Local Mode (Self-hosted)

Run Mem0 entirely on your local machine with no external API calls. Uses a local LLM (e.g., SGLang) and local embedding model.

**Requirements**:
- Local LLM server (e.g., SGLang at `http://localhost:1234/v1`)
- Qdrant vector store (`pip install qdrant-client`)
- Embedding model (e.g., `bge-large-zh-v1.5`)
- Python 3.11+ venv with mem0ai installed

**Setup**:

1. Install dependencies in a dedicated venv:
```bash
cd /path/to/your/mem0/dir
uv venv --python 3.11
source .venv/bin/activate
uv pip install mem0ai qdrant-client sentence-transformers
```

2. Start Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. Configure mem0.json:
```json
{
  "mode": "local",
  "mem0_server": "/path/to/your/mem0/dir/mem0_server.py",
  "mem0_python": "/path/to/your/mem0/dir/.venv/bin/python",
  "llm_base_url": "http://localhost:1234/v1",
  "llm_model": "qwen3",
  "embedder_model": "/path/to/your/bge-large-zh-v1.5",
  "embedding_dims": 1024,
  "qdrant_host": "localhost",
  "qdrant_port": 6333
}
```

4. Start the custom server:
```bash
cd /path/to/your/mem0/dir
python3 mem0_server.py search "test query" hermes-user 5 false
```

## Config

Config file: `$HERMES_HOME/mem0.json`

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | `platform` | `platform` or `local` |
| `user_id` | `hermes-user` | User identifier |
| `agent_id` | `hermes` | Agent identifier |
| `rerank` | `true` | Enable reranking for recall |
| `mem0_server` | *(required for local mode)* | Path to mem0_server.py |
| `mem0_python` | *(required for local mode)* | Path to Python executable in mem0 venv |
| `llm_base_url` | `http://localhost:1234/v1` | Local LLM endpoint |
| `llm_model` | `qwen3` | Local LLM model name |
| `embedder_model` | *(required for local mode)* | Embedding model path |
| `embedding_dims` | `1024` | Embedding dimension |
| `qdrant_host` | `localhost` | Qdrant host |
| `qdrant_port` | `6333` | Qdrant port |

**Environment variables** (alternative to mem0.json):
- `MEM0_SERVER` — path to mem0_server.py
- `MEM0_PYTHON` — path to Python executable in mem0 venv
- `LLM_BASE_URL` — local LLM endpoint
- `LLM_MODEL` — local LLM model name
- `EMBEDDER_MODEL` — embedding model path
- `EMBEDDING_DIMS` — embedding dimension
- `QDRANT_HOST` — Qdrant host
- `QDRANT_PORT` — Qdrant port

---

## What This Plugin Solves

### Problem 1: Conflicting Memories — stale values never decay

When a user changes a preference (e.g., "default browser is Chrome" → "now Firefox"), both memories coexist in the vector store with similar embeddings. Without conflict handling, the old memory keeps getting its access count bumped on every search, so it never decays — the agent sees contradictory facts every session.

**Before**: Both memories track access count equally → old value stays "immortal".

**After (asymmetric tracking)**: When a high-confidence conflict is detected, only the winner's access count is incremented. The loser is frozen (`track_frozen=True`) — its score decays naturally over 30-40 days until it falls below the cleanup threshold and is removed automatically.

### Problem 2: Similar but Not Conflicting Memories — false positives cause over-deduplication

Memories about the same topic but different aspects (e.g., "SGLang runs on port 1234" vs "SGLang uses Qwen3 model") have high embedding similarity. A naive dedup threshold would treat them as conflicts and suppress one.

**Before**: Single similarity threshold — either too aggressive (kills valid memories) or too lenient (misses real conflicts).

**After (dual thresholds + entity alignment)**: Three tiers of detection with different behaviors, plus entity-level verification for config-type memories. See [Conflict Detection & Resolution](#conflict-detection--resolution) for details.

---

## Conflict Detection & Resolution

### Dual-Threshold Similarity Tiers

Based on `bge-large-zh-v1.5` embedding cosine similarity:

| Similarity | Marker | Meaning | Tracking Behavior |
|-----------|--------|---------|-------------------|
| `> 0.97` + exact text match | — | Exact duplicate | Discarded (shadowed), never tracked |
| `> 0.97` + text differs | ⚠️ | High-confidence conflict | Asymmetric: only winner tracked |
| `0.92 – 0.97` | ⚠️ | High-confidence conflict | Asymmetric: only winner tracked |
| `0.75 – 0.92` | 🔗 | Possibly related | Both tracked normally |
| `< 0.75` | — | Unrelated | Normal injection |

**Key design**: The gap between 0.92 and 0.97 is intentional. Memories above 0.92 are treated as genuine conflicts (same fact, different value). Memories between 0.75-0.92 are flagged as related but NOT suppressed — they likely describe different aspects of the same topic.

### Entity Alignment for Config Conflicts

For configuration-type memories, embedding similarity alone is not enough — two config entries about different attributes may have high similarity but are not conflicting. The plugin extracts entity-value triples and checks a whitelist of mutually exclusive keys:

```python
_EXCLUSIVE_KEYS = {
    'ip', 'host', 'hostname', 'address', 'url', 'endpoint',
    'port', 'version', 'path', 'directory', 'dir',
    'username', 'password', 'token', 'api_key', 'secret',
    'default_browser', 'theme', 'language', 'os',
    'model', 'backend', 'engine', 'database',
}
```

When two memories share the same entity and an exclusive key type but have different values, the conflict is promoted to high-confidence (⚠️) even if embedding similarity is in the medium range (0.75–0.92). This catches detail-level config changes that pure cosine similarity might miss.

Example: "SGLang port is 8000" vs "SGLang port is 1234" — even if similarity is 0.82, the entity alignment detects `(sglang, port, 8000)` vs `(sglang, port, 1234)` and promotes it to ⚠️.

### Asymmetric Tracking

For high-confidence conflicts (⚠️):

1. **Winner selection**: Newer `updated_at` wins; tie-break by higher `access_count`
2. **Winner**: Access count incremented normally on each search hit
3. **Loser**: `track_frozen=True` — access count frozen, anchor compensation disabled
4. **Natural decay**: Loser's weighted score decays over ~30-40 days until it falls below `CLEANUP_THRESHOLD` (0.05) and is auto-removed

For medium-confidence related items (🔗): both memories track normally — no suppression.

Exact duplicates (shadowed): NOT touched — they decay naturally without being refreshed, preventing "immortal zombie" memories.

### Grace Period

Memories created within the last 14 days are exempt from asymmetric tracking — both sides track normally during this window. This prevents newly added memories from being immediately frozen by older ones, giving them time to accumulate access counts before conflict resolution kicks in.

### Conflict Resolution Rules (injected into system prompt)

When conflicting memories appear in the agent's context, these rules guide decision-making:

1. **时效优先**: Prefer the more recently updated memory (smaller "Updated N days ago")
2. **频次优先** (time diff < 3 days): Prefer higher access frequency (高频 > 中频 > 低频)
3. **冲突确认** (equal weight): Do not discard either side — mention both possibilities in the response
4. **上下文优先**: If a conflicting memory aligns with the current conversation context, override recency rules
5. **🔗 marker**: Related items are NOT conflicts — use context to determine if they coexist or replace each other

---

## Memory Decay System

The local mode includes a custom memory decay system that prevents offline time from being counted as "memory idle time".

### How It Works

- **Time Anchor**: A timestamp that freezes while the machine is offline
- **Effective Days**: `(anchor_time - last_accessed_at)` — measures actual system active time
- **Gap Compensation**: When the machine is offline for >36 hours, all `last_accessed_at` timestamps are shifted forward by the gap duration before updating the anchor

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HALF_LIFE_DAYS` | 7 | Half-life period for decay |
| `CLEANUP_THRESHOLD` | 0.05 | Minimum weighted score to keep |
| `ACCESS_COUNT_CAP` | 255 | Maximum access count |
| `GRACE_PERIOD` | 14 days | Memories created within grace period are protected |
| `GAP_THRESHOLD` | 36 hours | Offline duration that triggers timestamp shift |

### Decay Formula

```
weighted_score = min(access_count, CAP) × 0.5^(effective_days / half_life)
```

---

## Prefetch Optimization

To solve high-frequency memory occupation and improve recall quality:

- **Over-fetching**: Retrieves top_k=20 from Mem0, deduplicates, then injects only top 5 into the system prompt
- This ensures diverse results rather than the same high-frequency memories always dominating the top slots

## Circuit Breaker

After 5 consecutive API failures, the circuit breaker trips and pauses all Mem0 calls for 120 seconds. This prevents hammering a down server. The breaker resets automatically after cooldown, allowing a retry.

## Tools

| Tool | Description |
|------|-------------|
| `mem0_profile` | All stored memories about the user |
| `mem0_search` | Semantic search with optional reranking |
| `mem0_conclude` | Store a fact verbatim (no LLM extraction) |

## Troubleshooting

### FutureWarning from HuggingFace

If you see `FutureWarning: get_sentence_embedding_dimension is deprecated`, patch the embedding file:

```bash
sed -i 's/get_sentence_embedding_dimension/get_embedding_dimension/g' \
    /path/to/your/mem0/venv/lib/python3.11/site-packages/mem0/embeddings/huggingface.py
```

### Model Loading Warning on Every Call

This only happens once when `mem0_server.py` starts. The model weights are cached in `~/.cache/huggingface/`.

### Qdrant Connection Failed

Ensure Qdrant is running:
```bash
curl http://localhost:6333/collections
```

If it fails, restart Qdrant:
```bash
docker run -p 6333:6333 qdrant/qdrant
```
