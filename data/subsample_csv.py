"""
Take a stratified random subsample of a filepath,label CSV — same
bonafide/spoof ratio, smaller total size. Use this to get a fast
smoke-test dataset before scaling back up to the full one.
"""
import argparse, csv, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n", type=int, required=True,
                    help="total rows to keep (split proportionally by class)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.reader(f))

    bona = [r for r in rows if r[1].strip().lower() == "bonafide"]
    spoof = [r for r in rows if r[1].strip().lower() != "bonafide"]

    rng = random.Random(args.seed)
    rng.shuffle(bona)
    rng.shuffle(spoof)

    frac = args.n / len(rows)
    n_bona = max(1, int(len(bona) * frac))
    n_spoof = max(1, args.n - n_bona)

    kept = bona[:n_bona] + spoof[:n_spoof]
    rng.shuffle(kept)

    out_path = args.csv_path.rsplit(".", 1)[0] + f".small{args.n}.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(kept)
    print(f"[write] {out_path}: {len(kept)} rows "
          f"(bonafide={n_bona} spoof={n_spoof}) from {len(rows)} total")


if __name__ == "__main__":
    main()