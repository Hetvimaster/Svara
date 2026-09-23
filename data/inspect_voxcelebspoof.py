"""
Peek at ONE example from VoxCelebSpoof before downloading anything.
"""
from datasets import load_dataset

ds = load_dataset("MattyB95/VoxCelebSpoof", split="train", streaming=True)
example = next(iter(ds))
for k, v in example.items():
    if k == "audio" and isinstance(v, dict):
        print(f"{k}: dict with keys {list(v.keys())}, "
              f"sampling_rate={v.get('sampling_rate')}, "
              f"array_len={len(v.get('array', []))}")
    else:
        print(f"{k}: {repr(v)[:200]}")