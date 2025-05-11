import importlib, pkgutil
import utils.metrics

_METRICS: dict[str, type] = {}

def register(name: str):
    def deco(cls):
        _METRICS[name] = cls
        return cls
    return deco

def get_all() -> dict[str, type]:
    for _, module_name, _ in pkgutil.iter_modules(utils.metrics.__path__):
        importlib.import_module(f"utils.metrics.{module_name}")
    return _METRICS