import json, time
from datetime import datetime

def ensure_dirs(*paths):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)

def now_tag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def dump_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        self.dt = time.perf_counter() - self.t0
