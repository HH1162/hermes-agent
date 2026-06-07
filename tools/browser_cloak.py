"""CloakBrowser RPC backend — direct WebSocket JSON-RPC to CloakBridge daemon.

Routes browser tools through the local CloakBridge daemon via WebSocket RPC,
bypassing the agent-browser CLI (which sends CDP commands that don't trigger
Vue/Element UI event chains properly).

This module is used when ``browser.cloud_provider`` is set to ``cloak`` in
config.yaml.  It connects to the daemon's WebSocket URL returned by the
CloakBrowserProvider and sends JSON-RPC method calls that map directly to
the daemon's ``CloakSession`` methods.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

import websockets.sync.client as ws_sync

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session tracking — no local cache; trust _active_sessions as source of truth
# ---------------------------------------------------------------------------

def _find_live_daemon() -> str:
    """Discover an already-running CloakBridge daemon without spawning a new one.

    This is a fallback for when _active_sessions cache is stale (e.g. daemon
    was manually killed or crashed and restarted with a different port).
    Scans session info files and live daemon.py processes to find the port.
    """
    import glob as _glob
    import os as _os
    import json as _json
    import socket as _socket

    INFO_DIR = str(Path.home() / ".hermes" / "plugins" / "browser" / "cloak")

    # Strategy 1: check existing session info files (sorted by mtime, newest first)
    session_files = sorted(
        _glob.glob(_os.path.join(INFO_DIR, "session-*.json")),
        key=lambda p: _os.path.getmtime(p), reverse=True
    )
    for fpath in session_files:
        try:
            with open(fpath) as f:
                info = _json.load(f)
            ws_url = info.get("ws_url", "")
            if ws_url:
                parsed = urllib.parse.urlparse(ws_url)
                try:
                    with _socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
                        logger.info("browser/cloak: found live daemon via %s -> %s", fpath, ws_url)
                        return ws_url
                except OSError:
                    pass
        except Exception as e:
            logger.debug("browser/cloak: failed to read %s: %s", fpath, e)

    # Strategy 2: find daemon.py process via pgrep, read its env
    import subprocess as _sub
    try:
        out = _sub.run(["pgrep", "-f", "daemon.py", "-a"], capture_output=True, text=True, timeout=2)
        for line in out.stdout.strip().split("\n"):
            if "daemon.py" not in line:
                continue
            pid = line.split()[0]
            try:
                with open(f"/proc/{pid}/environ", "rb") as ef:
                    environ = ef.read().decode("utf-8", errors="replace")
                for entry in environ.split("\x00"):
                    if entry.startswith("CLOAKBRIDGE_INFO_PATH="):
                        env_file = entry.split("=", 1)[1]
                        if _os.path.isfile(env_file):
                            with open(env_file) as f:
                                info = _json.load(f)
                            ws_url = info.get("ws_url", "")
                            if ws_url:
                                parsed = urllib.parse.urlparse(ws_url)
                                try:
                                    with _socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
                                        logger.info("browser/cloak: found live daemon via pid %s -> %s", pid, ws_url)
                                        return ws_url
                                except OSError:
                                    pass
            except PermissionError:
                logger.debug("browser/cloak: cannot read /proc/%s/environ (permission denied)", pid)
            except Exception as e:
                logger.debug("browser/cloak: failed to check pid %s: %s", pid, e)
    except Exception as e:
        logger.debug("browser/cloak: pgrep failed: %s", e)

    # Strategy 3 (★ NEW): scan /tmp for cloak info files (daemon may write there)
    for fpath in _glob.glob("/tmp/cloak-info-*.json"):
        try:
            with open(fpath) as f:
                info = _json.load(f)
            ws_url = info.get("ws_url", "")
            if ws_url:
                parsed = urllib.parse.urlparse(ws_url)
                try:
                    with _socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
                        logger.info("browser/cloak: found live daemon via /tmp/%s -> %s",
                                   Path(fpath).name, ws_url)
                        return ws_url
                except OSError:
                    pass
        except Exception:
            pass

    return ""


def _get_session(task_id: Optional[str] = None) -> Dict[str, Any]:
    """Return CloakBrowser RPC session info. Always queries _active_sessions for the latest port.

    No local cache — _active_sessions in browser_tool.py is the single source of truth,
    and it already includes _probe_local_ws_port liveness checks.

    FALLBACK: if the cached daemon port is dead, scans for a live daemon
    process to recover from stale cache (daemon crash/restart with new port).
    """
    from tools.browser_tool import _get_session_info  # noqa: PLC0415

    effective_task_id = f"{task_id or 'default'}"
    session_info = _get_session_info(effective_task_id)
    ws_url = session_info.get("cdp_url", "")

    # ★ DEBUG: write to file to trace the actual path
    try:
        with open("/tmp/cloak-debug.log", "a") as debug_f:
            import time as _t
            debug_f.write(f"{_t.strftime('%H:%M:%S')} GET_SESSION task={effective_task_id} cached_url={ws_url}\n")
    except Exception:
        pass

    if ws_url:
        # Check if the cached daemon is actually alive
        parsed = urllib.parse.urlparse(ws_url)
        try:
            import socket as _socket
            with _socket.create_connection((parsed.hostname, parsed.port), timeout=0.5):
                return {"ws_url": ws_url}  # Daemon alive — use cached URL
        except OSError:
            logger.warning("browser/cloak: cached daemon at %s is dead, discovering new one", ws_url)

    # Fallback: discover live daemon (either fresh or from different task_id session)
    ws_url = _find_live_daemon()
    if ws_url:
        logger.info("browser/cloak: recovered lost daemon session -> %s", ws_url)
        return {"ws_url": ws_url}

    raise RuntimeError("CloakBrowser: no WebSocket URL available")


def _drop_session(task_id: Optional[str] = None) -> None:
    """Remove a CloakBrowser RPC session."""
    pass  # No local cache to manage; _active_sessions handles lifecycle.


# ---------------------------------------------------------------------------
# WebSocket RPC client (sync wrapper around async websockets)
# ---------------------------------------------------------------------------

def _rpc(session: Dict[str, Any], method: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
    """Send a JSON-RPC request to the CloakBridge daemon and return the result.

    Uses synchronous websockets to stay compatible with the browser tool interface
    which expects blocking calls.

    On connection failure, retries once — session is fetched fresh from
    _active_sessions each time (no stale cache risk).

    NOTE: timeout must exceed the daemon's internal budget to avoid premature cutoff:
    - click_ref budget: 30s → RPC timeout: 60s
    - type_text budget: 45s → RPC timeout: 90s
    """
    if timeout is None:
        timeout = 90 if method == "type" else 60
    message = {
        "method": method,
        "params": params or {},
        "id": 1
    }

    for attempt in range(2):
        ws_url = session.get("ws_url")

        if not ws_url:
            return {"success": False, "error": "No WebSocket URL available"}

        logger.info("CLOAK_RPC -> %s method=%s params_keys=%s (attempt %d)",
                     ws_url, method, list((params or {}).keys()), attempt + 1)

        try:
            with ws_sync.connect(ws_url, open_timeout=5) as ws:
                ws.send(json.dumps(message))
                response = json.loads(ws.recv(timeout=timeout))

            if "error" in response and response["error"]:
                logger.error("CLOAK_RPC <- %s method=%s ERROR: %s", ws_url, method, response["error"])
                return {
                    "success": False,
                    "error": response["error"].get("message", "Unknown RPC error")
                }

            logger.info("CLOAK_RPC <- %s method=%s OK", ws_url, method)
            return {
                "success": True,
                "data": response.get("result", {})
            }
        except Exception as e:
            err_msg = str(e)
            if attempt == 1:
                logger.error("CLOAK_RPC EXCEPTION (final): method=%s err=%s", method, err_msg)
                return {
                    "success": False,
                    "error": f"RPC failed after 2 attempts: {err_msg}"
                }
            logger.warning("CLOAK_RPC connection failed: method=%s err=%s — will retry", method, err_msg)
    return {"success": False, "error": "RPC failed: all attempts exhausted"}


# ---------------------------------------------------------------------------
# Browser tool functions for CloakBrowser
# ---------------------------------------------------------------------------

def cloak_snapshot(full: bool = False, task_id: Optional[str] = None) -> str:
    """Take an accessibility snapshot of the current page."""
    try:
        session = _get_session(task_id)
        result = _rpc(session, "snapshot", {"full": full})

        if not result["success"]:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Snapshot failed")
            })

        data = result.get("data", "")
        if isinstance(data, dict):
            # Daemon returns parsed snapshot
            snapshot_text = data.get("snapshot", data) if isinstance(data, dict) else str(data)
        else:
            snapshot_text = str(data)

        # Parse refs from snapshot text
        refs = {}
        import re
        ref_pattern = re.compile(r'\[ref=(@e\d+)\]')
        for match in ref_pattern.finditer(snapshot_text):
            ref_id = match.group(1)
            refs[ref_id] = True

        return json.dumps({
            "success": True,
            "data": {
                "snapshot": snapshot_text,
                "refs": refs
            },
            "element_count": len(refs)
        })
    except Exception as e:
        logger.error("cloak_snapshot failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_click(ref: str, task_id: Optional[str] = None) -> str:
    """Click an element by ref via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        # Strip @ prefix if present
        clean_ref = ref.lstrip("@")

        result = _rpc(session, "click", {"ref": clean_ref})

        if result["success"]:
            return json.dumps({
                "success": True,
                "clicked": clean_ref
            })
        else:
            err = result.get("error", "Click failed")
            # ★ P3: detect auto-refresh signal and pass it to agent
            needs_snapshot = "auto-refreshed" in err or "consecutive failures" in err
            return json.dumps({
                "success": False,
                "error": err,
                "needs_snapshot": needs_snapshot
            })
    except Exception as e:
        logger.error("cloak_click failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_type(ref: str, text: str, task_id: Optional[str] = None) -> str:
    """Type text into an element by ref via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        # Strip @ prefix if present
        clean_ref = ref.lstrip("@")

        result = _rpc(session, "type", {"ref": clean_ref, "text": text})

        if result["success"]:
            return json.dumps({
                "success": True,
                "typed": text,
                "element": clean_ref
            })
        else:
            err = result.get("error", "Type failed")
            # ★ P3: detect auto-refresh signal and pass it to agent
            needs_snapshot = "auto-refreshed" in err or "consecutive failures" in err
            return json.dumps({
                "success": False,
                "error": err,
                "needs_snapshot": needs_snapshot
            })
    except Exception as e:
        logger.error("cloak_type failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_console(expression: Optional[str] = None, clear: bool = False, task_id: Optional[str] = None) -> str:
    """Get console output or evaluate expression via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        params = {}
        if expression:
            params["expression"] = expression
        if clear:
            params["clear"] = True

        result = _rpc(session, "console", params)

        if result["success"]:
            data = result.get("data", {})
            messages = data.get("messages", [])

            # ★ Bug #19 fix: Check for eval_error — daemon sets this when evaluate() raises
            eval_error = data.get("eval_error")
            if eval_error:
                return json.dumps({
                    "success": False,
                    "error": f"JS evaluation failed: {eval_error}",
                    "messages": messages,
                })

            # ★ Extract eval result from messages if expression was provided.
            # Daemon appends eval result as {"type": "eval", "text": "..."} in messages.
            eval_result = None
            if expression and messages:
                last_msg = messages[-1]
                if isinstance(last_msg, str):
                    try:
                        parsed = json.loads(last_msg)
                        if parsed.get("type") == "eval":
                            eval_result = parsed.get("text")
                    except (json.JSONDecodeError, AttributeError):
                        pass

            return json.dumps({
                "success": True,
                "messages": messages,
                "result": eval_result,
                "result_type": type(eval_result).__name__ if eval_result is not None else None,
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Console failed")
            })
    except Exception as e:
        logger.error("cloak_console failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_scroll(direction: str, task_id: Optional[str] = None) -> str:
    """Scroll the page via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        if direction not in ("up", "down"):
            return json.dumps({
                "success": False,
                "error": f"Invalid direction '{direction}'. Use 'up' or 'down'."
            })

        result = _rpc(session, "scroll", {"direction": direction})

        if result["success"]:
            return json.dumps({
                "success": True,
                "scrolled": direction
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Scroll failed")
            })
    except Exception as e:
        logger.error("cloak_scroll failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_press(key: str, task_id: Optional[str] = None) -> str:
    """Press a keyboard key via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "press", {"key": key})

        if result["success"]:
            return json.dumps({
                "success": True,
                "pressed": key
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Press failed")
            })
    except Exception as e:
        logger.error("cloak_press failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_navigate(url: str, task_id: Optional[str] = None) -> str:
    """Navigate to a URL via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "navigate", {"url": url})

        if result["success"]:
            data = result.get("data", {})
            return json.dumps({
                "success": True,
                "url": data.get("url", url),
                "title": data.get("title", ""),
                "snapshot": ""
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Navigation failed")
            })
    except Exception as e:
        logger.error("cloak_navigate failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_close(task_id: Optional[str] = None) -> str:
    """Close the CloakBrowser session."""
    try:
        session = _get_session(task_id)
        _rpc(session, "close")
        _drop_session(task_id)
        return json.dumps({"success": True, "closed": True})
    except Exception as e:
        logger.error("cloak_close failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_back(task_id: Optional[str] = None) -> str:
    """Navigate back in browser history via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "back")

        if result["success"]:
            data = result.get("data", {})
            return json.dumps({
                "success": True,
                "url": data.get("url", ""),
                "title": data.get("title", "")
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Back navigation failed")
            })
    except Exception as e:
        logger.error("cloak_back failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_evaluate(expression: str, task_id: Optional[str] = None) -> str:
    """Evaluate JavaScript via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "evaluate", {"expression": expression})

        if result["success"]:
            return json.dumps({
                "success": True,
                "result": result.get("data")
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Evaluate failed")
            })
    except Exception as e:
        logger.error("cloak_evaluate failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_screenshot(task_id: Optional[str] = None) -> str:
    """Take a screenshot via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "screenshot")

        if result["success"]:
            # Daemon returns base64 string directly (not nested dict)
            data = result.get("data", "")
            if isinstance(data, dict):
                data = data.get("data", "")
            return json.dumps({
                "success": True,
                "data": data
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Screenshot failed")
            })
    except Exception as e:
        logger.error("cloak_screenshot failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_get_images(task_id: Optional[str] = None) -> str:
    """Get all images on the current page via CloakBridge daemon."""
    try:
        session = _get_session(task_id)

        result = _rpc(session, "get_images")

        if result["success"]:
            images = result.get("data", [])
            return json.dumps({
                "success": True,
                "images": images,
                "count": len(images)
            })
        else:
            return json.dumps({
                "success": False,
                "error": result.get("error", "Failed to get images")
            })
    except Exception as e:
        logger.error("cloak_get_images failed: %s", e)
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def cloak_vision(question: str, annotate: bool = False, task_id: Optional[str] = None) -> Any:
    """Take a screenshot via CloakBridge daemon and return vision analysis.

    NOTE: annotate is not supported in cloak mode (daemon doesn't support overlay).
    The screenshot is already compressed by the daemon (1280px + JPEG q=80).

    Returns dict when native vision is enabled (framework extracts image_data_url),
    or json string for fallback mode. Matches browser_vision return contract.
    """
    import base64
    import uuid as uuid_mod
    from pathlib import Path

    try:
        from hermes_constants import get_hermes_dir
        screenshots_dir = get_hermes_dir("cache/screenshots", "browser_screenshots")
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = screenshots_dir / f"browser_screenshot_{uuid_mod.uuid4().hex}.jpg"

        # Get screenshot from cloak daemon (already compressed)
        result = cloak_screenshot(task_id)
        res_data = json.loads(result)

        if not res_data.get("success"):
            return json.dumps({
                "success": False,
                "error": res_data.get("error", "Screenshot failed"),
                "note": "annotate is not supported in CloakBrowser mode" if annotate else ""
            })

        # Decode base64 and save to file
        b64_data = res_data.get("data", "")
        if not b64_data:
            return json.dumps({"success": False, "error": "Empty screenshot data"})

        img_bytes = base64.b64decode(b64_data)
        screenshot_path.write_bytes(img_bytes)

        data_url = f"data:image/jpeg;base64,{b64_data}"

        from tools.vision_tools import (
            _build_native_vision_tool_result,
            _should_use_native_vision_fast_path,
        )

        # ★ Native vision fast path — MUST return dict (not json.dumps'd).
        # The framework inspects the return type: dict triggers image injection,
        # string is treated as plain text. Original browser_vision does this at L3352.
        if _should_use_native_vision_fast_path():
            native_result = _build_native_vision_tool_result(
                image_url=str(screenshot_path),
                question=question,
                image_data_url=data_url,
                image_size_bytes=len(img_bytes),
            )
            meta = native_result.setdefault("meta", {})
            meta["screenshot_path"] = str(screenshot_path)
            native_result["text_summary"] = (
                f"{native_result.get('text_summary', '')} "
                f"Screenshot path: {screenshot_path}"
            ).strip()
            return native_result  # ★ Return dict, NOT json.dumps'd

        # Fallback: return screenshot path for manual inspection
        return json.dumps({
            "success": True,
            "analysis": f"Screenshot captured ({len(img_bytes)} bytes). Question: {question}. "
                        f"Use browser_snapshot for text-based page analysis.",
            "screenshot_path": str(screenshot_path),
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("cloak_vision failed: %s", e, exc_info=True)
        return json.dumps({
            "success": False,
            "error": str(e)
        })
