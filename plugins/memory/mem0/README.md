# Mem0 Memory Provider

Server-side LLM fact extraction with semantic search, reranking, and automatic deduplication.

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
echo "MEM0_API_KEY=your-key" >> ~/.hermes/.env
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
cd /media/data/mem0
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
  "llm_base_url": "http://localhost:1234/v1",
  "llm_model": "qwen3",
  "embedder_model": "/home/herocco/bge/bge-large-zh-v1.5",
  "embedding_dims": 1024,
  "qdrant_host": "localhost",
  "qdrant_port": 6333
}
```

4. Start the custom server:
```bash
cd /media/data/mem0
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
| `llm_base_url` | `http://localhost:1234/v1` | Local LLM endpoint |
| `llm_model` | `qwen3` | Local LLM model name |
| `embedder_model` | `/home/herocco/bge/bge-large-zh-v1.5` | Embedding model path |
| `embedding_dims` | `1024` | Embedding dimension |
| `qdrant_host` | `localhost` | Qdrant host |
| `qdrant_port` | `6333` | Qdrant port |

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
    /media/data/mem0/.venv/lib/python3.11/site-packages/mem0/embeddings/huggingface.py
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
