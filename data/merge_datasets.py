import argparse, csv

def read_rows(path):
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
        rows = []
        for src in sources:
            rows.extend(read_rows(src))
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[merge] {out_path}: {len(rows)} rows from {len(sources)} sources")

if __name__ == "__main__":
    main()