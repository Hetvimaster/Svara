"""
Corrected inspection: disables audio decoding so we can see the file path
(needed to tell Clone/ [spoof] apart from Chunk/ [bonafide]), and samples
a spread of rows instead of just the first one so we can see the real
folder distribution before writing any build logic.
"""
from collections import Counter
from datasets import load_dataset

N_PEEK = 200  # how many rows to sample for the folder-distribution check

ds = load_dataset("MattyB95/VoxCelebSpoof", split="train", streaming=True)
ds = ds.decode(False)  # keep raw {"path": ..., "bytes": ...} instead of decoding audio

top_folders = Counter()
exts = Counter()
first_example = None

for i, ex in enumerate(ds):
    if i >= N_PEEK:
        break
    if first_example is None:
        first_example = ex
        print("[first example] keys:", list(ex.keys()))
        for k, v in ex.items():
            if k == "audio":
                print(f"  audio.path = {v.get('path')!r}")
                print(f"  audio.bytes = {'<bytes present>' if v.get('bytes') else None}")
            else:
                print(f"  {k} = {repr(v)[:200]}")

    path = ex["audio"].get("path") or ""
    top = path.split("/")[0] if "/" in path else path
    top_folders[top] += 1
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else "?"
    exts[ext] += 1

print(f"\n[folder distribution over {N_PEEK} sampled rows]")
for folder, count in top_folders.most_common():
    print(f"  {folder:25s} {count}")

print(f"\n[file extensions]")
for ext, count in exts.most_common():
    print(f"  .{ext:10s} {count}")