import argparse, csv, os

def read_rows(path):
    if not os.path.isfile(path):
        print(f"[warn] skipping missing source: {path}")
        return []
    with open(path, newline="") as f:
        return list(csv.reader(f))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_sources", nargs="+", required=True)
    ap.add_argument("--val_sources", nargs="+", required=True)
    ap.add_argument("--out_train", default="data/combined/train_v2.csv")
    ap.add_argument("--out_val", default="data/combined/val_v2.csv")
    args = ap.parse_args()

    for sources, out_path in [(args.train_sources, args.out_train), (args.val_sources, args.out_val)]:
        rows, used = [], 0
        for src in sources:
            r = read_rows(src)
            if r:
                used += 1
            rows.extend(r)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        n_bona = sum(1 for row in rows if len(row) == 2 and row[1].strip().lower() == "bonafide")
        print(f"[merge] {out_path}: {len(rows)} rows from {used}/{len(sources)} sources "
              f"(bonafide={n_bona} spoof={len(rows)-n_bona})")

if __name__ == "__main__":
    main()