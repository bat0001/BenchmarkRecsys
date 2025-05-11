import importlib, pkgutil
import utils.formatters

_FORMATTERS: dict[str, type] = {}
_loaded = False

def register(name: str):
    def deco(cls):
        _FORMATTERS[name.upper()] = cls()
        return cls
    return deco

def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    for finder, module_name, _ in pkgutil.iter_modules(utils.formatters.__path__):
        importlib.import_module(f"utils.formatters.{module_name}")
    _loaded = True

def get(name: str):
    _ensure_loaded()
    return _FORMATTERS[name.upper()]