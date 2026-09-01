PAIR_BASE = 1 << 20

def _pack(a, b):
    return a * PAIR_BASE + b

def _unpack(p):
    return divmod(p, PAIR_BASE)
