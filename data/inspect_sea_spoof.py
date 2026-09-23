"""
Build filepath,label CSVs from Jack-ppkdczgx/SEA-Spoof (streaming, decoded
audio -> written to disk as wav). Prints the schema of the first example so
you can see if the auto-guess was right; override with --label_column if not.
"""
import argparse, csv, os, random
from datasets import load_dataset
import soundfile as sf
import numpy as np

SPOOF_HINTS = ("spoof", "fake", "synth", "tts", "clone", "generated")
BONA_HINTS = ("bonafide", "bona", "real", "human", "genuine")


def guess_label(example, label_column):
    if label_column and label_column in example:
        raw = str(example[label_column]).strip().lower()
    else:
        raw = None
        for k, v in example.items():
            if k == "audio":
                continue
            kv = f"{k}={v}".lower()
            if any(h in kv for h in SPOOF_HINTS + BONA_HINTS):
                raw = str(v).strip().lower()
                break
        if raw is None:
            return None
    if any(h in raw for h in SPOOF_HINTS) or raw in ("0", "fake"):
        return "spoof"
    if any(h in raw for h in BONA_HINTS) or raw in ("1", "real"):
        return "bonafide"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/sea_spoof_cache")
    ap.add_argument("--out_train", default="data/sea_spoof_train.csv")
    ap.add_argument("--out_val", default="data/sea_spoof_val.csv")
    ap.add_argument("--max_per_class", type=int, default=150)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--label_column", default=None,
                     help="exact column name from inspect_sea_spoof.py, e.g. 'label'. Leave unset to auto-detect.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    ds = load_dataset("Jack-ppkdczgx/SEA-Spoof", split="train", streaming=True)

    bona_rows, spoof_rows = [], []
    checked_schema = False

    for ex in ds:
        if len(bona_rows) >= args.max_per_class and len(spoof_rows) >= args.max_per_class:
            break
        if not checked_schema:
            print("[schema] example keys:", [k for k in ex.keys() if k != "audio"])
            checked_schema = True

        label = guess_label(ex, args.label_column)
        if label is None:
            continue
        bucket = bona_rows if label == "bonafide" else spoof_rows
        if len(bucket) >= args.max_per_class:
            continue

        audio = ex["audio"]
        array, sr = audio.get("array"), audio.get("sampling_rate", 16000)
        if array is None or len(array) == 0:
            continue

        out_path = os.path.abspath(os.path.join(args.out_dir, f"{label}_{len(bucket):04d}.wav"))
        sf.write(out_path, np.asarray(array, dtype=np.float32), sr)
        bucket.append((out_path, label))
        if (len(bona_rows) + len(spoof_rows)) % 50 == 0:
            print(f"[progress] bonafide={len(bona_rows)} spoof={len(spoof_rows)}")

    print(f"[done] bonafide={len(bona_rows)} spoof={len(spoof_rows)}")
    if not bona_rows or not spoof_rows:
        raise SystemExit("[error] one class empty — run inspect_sea_spoof.py, check the real "
                          "label column/values, pass --label_column")

    all_rows = bona_rows + spoof_rows
    rng.shuffle(all_rows)
    cut = int(len(all_rows) * (1 - args.val_fraction))
    train_rows, val_rows = all_rows[:cut], all_rows[cut:]
    for rows, out_path in [(train_rows, args.out_train), (val_rows, args.out_val)]:
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        n_bona = sum(1 for _, l in rows if l == "bonafide")
        print(f"[write] {out_path}: {len(rows)} rows (bonafide={n_bona} spoof={len(rows)-n_bona})")


if __name__ == "__main__":
    main()