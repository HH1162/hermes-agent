#!/usr/bin/env python3
"""Mem0 memory lifecycle management server — reference implementation.

This script adds access frequency tracking, exponential decay scoring,
and automated cleanup of stale memories on top of the Mem0 SDK.

Usage:
    python mem0_server.py <action> [args...]

Actions:
    search <query> <user_id> [top_k] [rerank]    # Search memories (auto-tracks access)
    add <json_messages> <user_id> <agent_id>     # Add new memory
    get_all <user_id>                            # Get all memories
    profile <user_id>                            # Get user profile memories
    stats [user_id]                              # Show access statistics
    least_used [user_id] [top_n]                 # Show least-used memories
    cleanup [--dry-run] [--threshold X] [user_id] # Remove stale memories below threshold

Cleanup config:
    Default threshold: 0.05 (weighted_score)
    Half-life: 7 days
    Grace period: newly created memories (< 14 days old) are protected
    No hard protection by access count — exponential decay handles all cases
"""

import json
import os
import sys
from datetime import datetime, timezone

from mem0 import Memory
from qdrant_client import QdrantClient

# --- Tunable parameters ---

HALF_LIFE_DAYS = 7.0       # Exponential decay half-life (days)
CLEANUP_THRESHOLD = 0.05   # Memories with weighted_score < this are candidates for deletion
ACCESS_COUNT_CAP = 255     # Hard cap on access_count — prevents unbounded growth

# --- Time anchor: freezes decay while machine is offline ---

STATE_FILE = os.path.expanduser("~/.hermes/mem0_state.json")


def get_anchor_time():
    """Get the system's last-active timestamp (time anchor).

    This is the reference point for all decay calculations.
    While the machine is offline, this timestamp does not advance,
    effectively freezing memory decay during downtime.

    Includes sanity check: if the stored anchor or system clock is
    before MIN_VALID_YEAR, it is considered corrupted (e.g. CMOS
    battery failure, NTP desync) and a safe fallback is used.
    """
    MIN_VALID_YEAR = 2024

    try:
        if os.path.exists(STATE_FILE):
            state = json.loads(open(STATE_FILE).read())
            ts = state.get("last_active", "")
            if ts:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt.year >= MIN_VALID_YEAR:
                    return dt
    except Exception:
        pass

    now = datetime.now(timezone.utc)
    if now.year >= MIN_VALID_YEAR:
        return now
    # Extreme fallback: system clock itself is broken (e.g. CMOS dead)
    return datetime(MIN_VALID_YEAR, 1, 1, tzinfo=timezone.utc)


def update_anchor_time():
    """Called on search/add to advance the time anchor.

    Only invoked when the user is actively interacting with the system.
    During downtime, this is never called, so the anchor stays frozen.
    """
    try:
        state = {"last_active": datetime.now(timezone.utc).isoformat()}
        tmp = STATE_FILE + ".tmp"
        open(tmp, 'w').write(json.dumps(state, indent=2))
        os.replace(tmp, STATE_FILE)
    except Exception:
        pass


def compute_weighted_score(access_count, last_accessed_iso, anchor_time=None):
    """Exponential decay based on system active time, not physical wall-clock time.

    score = min(access_count, CAP) * 0.5^(effective_days / half_life)

    effective_days = (anchor_time - last_accessed_at)

    When anchor_time is None, defaults to current wall-clock time.
    During cleanup/stats, anchor_time is explicitly passed to ensure
    consistent scoring across all memories in a single run.

    Timestamp parsing uses Python native fromisoformat + tzinfo check for
    robust handling of UTC, timezone-aware, naive, Z-suffix, and negative-offset
    ISO strings. On parse failure, returns 0.0 (expired) to avoid immortal zombies.

    Examples (half_life=7 days):
    - 3 accesses today -> 3.0
    - 10 accesses, last seen 21 days ago -> 1.25
    - 1 access, last seen 33 days ago -> ~0.05 (cleanup threshold)
    """
    if not access_count or not last_accessed_iso or last_accessed_iso == 'never':
        return 0.0
    try:
        ts = last_accessed_iso.replace('Z', '+00:00')
        last_dt = datetime.fromisoformat(ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)

        # Use anchor_time (last system activity) instead of wall-clock now()
        # This freezes decay while the machine is offline
        if anchor_time is None:
            anchor_time = get_anchor_time()
        elif anchor_time.tzinfo is None:
            anchor_time = anchor_time.replace(tzinfo=timezone.utc)

        days = max(0, (anchor_time - last_dt).total_seconds() / 86400)
        return float(min(access_count, ACCESS_COUNT_CAP)) * (0.5 ** (days / HALF_LIFE_DAYS))
    except Exception:
        # Timestamp parse failure -> treat as expired to avoid immortal zombies
        return 0.0


def get_memory_client():
    """Create Memory client with local config.

    Replace the values below with your own deployment settings.
    See https://docs.mem0.ai/quick-start for full configuration options.
    """
    config = {
        'llm': {
            'provider': 'openai',
            'config': {
                'api_key': 'your-api-key-here',
                'openai_base_url': 'http://localhost:1234/v1',
                'model': 'qwen3'
            }
        },
        'embedder': {
            'provider': 'huggingface',
            'config': {
                'model': '/path/to/embedding-model'
            }
        },
        'vector_store': {
            'provider': 'qdrant',
            'config': {
                'collection_name': 'mem0',
                'embedding_model_dims': 1024,
                'host': 'localhost',
                'port': 6333
            }
        },
        'custom_instructions': """
## Storage Rules (Highest Priority)

This is a working AI assistant. Only store information that retains value across sessions.

STORE:
- User preferences, habits, style requirements
- Iron rules, red lines, taboos
- Environment facts: tool paths, service ports, venv locations
- Technical decisions and architecture choices
- Verified stable patterns and pitfalls
- User identity and project information

DO NOT STORE:
- Single-session task progress or intermediate state
- Code modification logs (belong in skills, not memory)
- One-time debugging conclusions (unless verified as a stable pattern)
- Emotional descriptions ("nervous", "frustrated")
- Temporary file paths, commit SHAs, PR numbers, branch names
- Content already fully covered by USER.md or skills
- Speculation or unconfirmed hypotheses
"""
    }
    return Memory.from_config(config)


def get_access_payload(qdrant, mem_id):
    """Fetch access_count and last_accessed_at for a memory from Qdrant.

    NOTE: This is kept for backward compatibility but is SLOW (N+1 queries).
    Prefer batch_get_access_payload() for bulk operations (stats, cleanup, least_used).
    """
    try:
        pt = qdrant.retrieve(collection_name='mem0', ids=[str(mem_id)], with_payload=True)
        if pt:
            p = pt[0].payload
            return p.get('access_count', 0), p.get('last_accessed_at', 'never')
    except Exception:
        pass
    return 0, 'never'


def batch_get_access_payload(qdrant, mem_ids):
    """Batch retrieve access payloads for multiple memory IDs.

    Replaces N sequential Qdrant queries with a single batch retrieve call.
    Returns dict: {mem_id: (access_count, last_accessed_at)}

    This eliminates the N+1 query problem in stats/cleanup/least_used operations.
    Without this, 100 memories = 100 network roundtrips (1-3 seconds).
    With this, 100 memories = 1 network roundtrip (<50ms).
    """
    if not mem_ids:
        return {}

    try:
        pts = qdrant.retrieve(
            collection_name='mem0',
            ids=[str(mid) for mid in mem_ids],
            with_payload=True
        )
        result = {}
        for p in pts:
            payload = p.payload or {}
            result[str(p.id)] = (
                payload.get('access_count', 0),
                payload.get('last_accessed_at', 'never')
            )
        return result
    except Exception:
        return {}


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: mem0_server.py <action> [args...]"}))
        sys.exit(1)

    action = sys.argv[1]

    # Advance time anchor on user-facing actions (search/add)
    if action in ("search", "add"):
        update_anchor_time()

    try:
        client = get_memory_client()

        if action == "search":
            query = sys.argv[2]
            user_id = sys.argv[3]
            top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 10
            rerank = sys.argv[5] == "true" if len(sys.argv) > 5 else False

            results = client.search(
                query=query,
                filters={'user_id': user_id},
                top_k=top_k,
                rerank=rerank,
                threshold=0.4
            )

            # Track access frequency for each matched memory
            if results:
                results_list = results.get('results', []) if isinstance(results, dict) else results
                if results_list:
                    qdrant = QdrantClient(host='localhost', port=6333)
                    now = datetime.now(timezone.utc).isoformat()
                    for r in results_list:
                        mem_id = str(r.get('id', ''))
                        if mem_id:
                            try:
                                ac, _ = get_access_payload(qdrant, mem_id)
                                qdrant.set_payload(
                                    collection_name='mem0',
                                    payload={
                                        'access_count': min(ac + 1, ACCESS_COUNT_CAP),
                                        'last_accessed_at': now
                                    },
                                    points=[mem_id]
                                )
                            except Exception:
                                pass

            print(json.dumps(results))

        elif action == "add":
            messages_json = sys.argv[2]
            user_id = sys.argv[3]
            agent_id = sys.argv[4]

            messages = json.loads(messages_json)
            client.add(messages, user_id=user_id, agent_id=agent_id)
            print(json.dumps({"result": "Fact stored."}))

        elif action == "get_all":
            user_id = sys.argv[2]
            memories = client.get_all(filters={'user_id': user_id})
            print(json.dumps(memories))

        elif action == "profile":
            user_id = sys.argv[2]
            memories = client.get_all(filters={'user_id': user_id})
            if isinstance(memories, dict):
                memories = memories.get("results", [])
            lines = [m.get("memory", "") for m in memories if m.get("memory")]
            print(json.dumps({
                "result": "\n".join(lines),
                "count": len(lines)
            }))

        elif action == "stats":
            user_id = sys.argv[2] if len(sys.argv) > 2 else "example-user"
            qdrant = QdrantClient(host='localhost', port=6333)

            # Use time anchor for consistent scoring (not wall-clock now)
            anchor = get_anchor_time()

            all_mems = client.get_all(filters={'user_id': user_id})
            results = all_mems.get('results', []) if isinstance(all_mems, dict) else []

            # BATCH retrieve all payloads in one call (eliminates N+1 query problem)
            mem_ids = [str(m.get('id', '')) for m in results if m.get('id')]
            payload_map = batch_get_access_payload(qdrant, mem_ids)

            total = len(results)
            with_access = 0
            never_accessed = 0
            max_access = 0
            total_access = 0
            total_weighted = 0.0
            max_weighted = 0.0

            for m in results:
                mem_id = str(m.get('id', ''))
                ac, la = payload_map.get(mem_id, (0, 'never'))
                total_access += ac
                if ac > 0:
                    with_access += 1
                else:
                    never_accessed += 1
                if ac > max_access:
                    max_access = ac

                w = compute_weighted_score(ac, la, anchor_time=anchor)
                total_weighted += w
                max_weighted = max(max_weighted, w)

            avg_access = total_access / with_access if with_access > 0 else 0
            avg_weighted = total_weighted / with_access if with_access > 0 else 0

            print(json.dumps({
                'total_memories': total,
                'never_accessed': never_accessed,
                'with_access': with_access,
                'avg_access_raw': round(avg_access, 2),
                'max_access_raw': max_access,
                'total_accesses': total_access,
                'avg_weighted_score': round(avg_weighted, 4),
                'max_weighted_score': round(max_weighted, 4),
                'half_life_days': int(HALF_LIFE_DAYS)
            }))

        elif action == "least_used":
            user_id = sys.argv[2] if len(sys.argv) > 2 else "example-user"
            top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            qdrant = QdrantClient(host='localhost', port=6333)

            anchor = get_anchor_time()

            all_mems = client.get_all(filters={'user_id': user_id})
            results = all_mems.get('results', []) if isinstance(all_mems, dict) else []

            # BATCH retrieve all payloads in one call (eliminates N+1 query problem)
            mem_ids = [str(m.get('id', '')) for m in results if m.get('id')]
            payload_map = batch_get_access_payload(qdrant, mem_ids)

            with_stats = []
            for m in results:
                mem_id = str(m.get('id', ''))
                ac, la = payload_map.get(mem_id, (0, 'never'))
                w = compute_weighted_score(ac, la, anchor_time=anchor)
                with_stats.append({
                    'id': mem_id,
                    'memory': m.get('memory', '')[:120],
                    'access_count': ac,
                    'last_accessed_at': la,
                    'weighted_score': round(w, 4),
                    'created_at': m.get('created_at', '')
                })

            with_stats.sort(key=lambda x: x['weighted_score'])
            print(json.dumps(with_stats[:top_n], indent=2))

        elif action == "cleanup":
            # Parse arguments: cleanup [--dry-run] [--threshold X] [user_id]
            args = sys.argv[2:]
            dry_run = '--dry-run' in args
            user_id = 'example-user'
            threshold = CLEANUP_THRESHOLD

            # Parse optional threshold
            for i, arg in enumerate(args):
                if arg == '--threshold' and i + 1 < len(args):
                    threshold = float(args[i + 1])
                elif arg not in ('--dry-run', '--threshold') and not arg.startswith('-'):
                    user_id = arg

            qdrant = QdrantClient(host='localhost', port=6333)

            anchor = get_anchor_time()

            all_mems = client.get_all(filters={'user_id': user_id})
            results = all_mems.get('results', []) if isinstance(all_mems, dict) else []

            # BATCH retrieve all payloads in one call (eliminates N+1 query problem)
            mem_ids = [str(m.get('id', '')) for m in results if m.get('id')]
            payload_map = batch_get_access_payload(qdrant, mem_ids)

            candidates = []
            kept = []

            for m in results:
                mem_id = str(m.get('id', ''))
                ac, la = payload_map.get(mem_id, (0, 'never'))
                w = compute_weighted_score(ac, la, anchor_time=anchor)

                # Grace period: newly created memories (< 14 days old) are protected.
                # Uses anchor_time as single source of truth instead of datetime.now()
                # to remain resilient against system clock drift (NTP desync, CMOS failure).
                grace_protected = False
                created_at = m.get('created_at', '')
                if created_at:
                    try:
                        ca_ts = created_at.replace('Z', '+00:00')
                        ca_dt = datetime.fromisoformat(ca_ts)
                        if ca_dt.tzinfo is None:
                            ca_dt = ca_dt.replace(tzinfo=timezone.utc)
                        days_old = (anchor - ca_dt).total_seconds() / 86400
                        grace_protected = days_old < 14
                    except Exception:
                        # If timestamp parsing fails, default to protecting the memory
                        # to avoid accidentally deleting newly created entries
                        grace_protected = True

                entry = {
                    'id': mem_id,
                    'memory': m.get('memory', '')[:120],
                    'access_count': ac,
                    'weighted_score': round(w, 4),
                    'last_accessed_at': la,
                    'grace_protected': grace_protected
                }

                if w < threshold and not grace_protected:
                    candidates.append(entry)
                else:
                    entry['reason'] = 'grace_period' if grace_protected else f'score {w:.4f} >= threshold {threshold}'
                    kept.append(entry)

            # Sort candidates by weighted score (lowest first)
            candidates.sort(key=lambda x: x['weighted_score'])

            if dry_run:
                output = {
                    'mode': 'DRY_RUN',
                    'threshold': threshold,
                    'half_life_days': int(HALF_LIFE_DAYS),
                    'to_delete': len(candidates),
                    'kept': len(kept),
                    'candidates': candidates
                }
                print(json.dumps(output, indent=2))
            else:
                # Delete via Mem0 SDK (synchronizes both metadata layer AND Qdrant vector store)
                deleted_ids = []
                failed_ids = []
                for c in candidates:
                    try:
                        client.delete(memory_id=c['id'])
                        deleted_ids.append(c['id'])
                    except Exception as e:
                        failed_ids.append({'id': c['id'], 'error': str(e)})

                output = {
                    'mode': 'EXECUTED',
                    'threshold': threshold,
                    'deleted_count': len(deleted_ids),
                    'deleted_ids': deleted_ids,
                    'failed': failed_ids,
                    'kept_count': len(kept)
                }
                print(json.dumps(output, indent=2))

        else:
            print(json.dumps({"error": f"Unknown action: {action}"}))
            sys.exit(1)

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
