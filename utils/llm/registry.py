_JUDGES = {}

def register(name):
    def deco(cls):
        _JUDGES[name.lower()] = cls
        return cls
    return deco

def get_judge(name="hf"):
    return _JUDGES[name.lower()]