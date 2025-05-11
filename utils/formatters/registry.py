_FORMATTERS = {}

def register(name):
    def deco(cls):
        _FORMATTERS[name] = cls()
        return cls
    return deco

def get(name):
    return _FORMATTERS[name]