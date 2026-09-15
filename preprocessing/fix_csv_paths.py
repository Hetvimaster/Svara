import csv
import os

ROOT = "/Users/yashcomputers/Desktop/Svara"

for name in ["train.csv", "val.csv"]:
    path = os.path.join(ROOT, "data/combined", name)
    with open(path, newline="") as f:
        rows = [(os.path.join(ROOT, p), label) for p, label in csv.reader(f) if p]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"rewrote {len(rows)} rows in {name}")