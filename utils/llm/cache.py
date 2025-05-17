"""
Very small disk‑backed cache to avoid re‑querying the LLM for the
*exact* same prompt.  Works for both online and offline runs.

Key  : md5(prompt)
Value: whatever you want to pickle (int, str, dict…)
"""
from __future__ import annotations
import hashlib, pickle, pathlib, threading

_CACHE_FILE = pathlib.Path("~/.gfn_llm_cache.pkl").expanduser()
_LOCK       = threading.Lock()

try:
    _MEM: dict[str, object] = pickle.loads(_CACHE_FILE.read_bytes())
except FileNotFoundError:
    _MEM = {}

def _key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf‑8")).hexdigest()

def get_cached(prompt: str):
    """Return cached value or None."""
    return _MEM.get(_key(prompt))

def set_cached(prompt: str, value):
    """Store value and flush to disk in a thread‑safe way."""
    k = _key(prompt)
    with _LOCK:
        _MEM[k] = value
        try:
            _CACHE_FILE.write_bytes(pickle.dumps(_MEM))
        except Exception:
            # Best‑effort: if disk write fails we just skip caching
            pass