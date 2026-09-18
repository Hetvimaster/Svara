"""
Print a handful of file paths per label from a CSV so you can manually
listen and confirm they're actually labeled correctly, before trusting
any EER number computed from them.
"""
import argparse, csv, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--n_per_label", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.reader(f))

    bona = [r[0] for r in rows if r[1].strip().lower() == "bonafide"]
    spoof = [r[0] for r in rows if r[1].strip().lower() != "bonafide"]

    rng = random.Random(args.seed)
    rng.shuffle(bona)
    rng.shuffle(spoof)

    print(f"--- {args.n_per_label} labeled BONAFIDE ---")
    for p in bona[:args.n_per_label]:
        print(" ", p)
    print(f"\n--- {args.n_per_label} labeled SPOOF ---")
    for p in spoof[:args.n_per_label]:
        print(" ", p)

if __name__ == "__main__":
    main()