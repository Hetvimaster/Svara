"""
Downsample the IndicSynth portion of the merged CSV, but always KEEP every
MLAAD row (identified by 'mlaad' in the filepath) since there are few of
them and they're the whole reason for this fine-tune round.
"""
import argparse, csv, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path")
    ap.add_argument("--indic_n", type=int, required=True,
                    help="how many non-MLAAD rows to keep (split proportionally by class)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = list(csv.reader(f))

    mlaad_rows = [r for r in rows if "mlaad" in r[0].lower()]
    indic_rows = [r for r in rows if "mlaad" not in r[0].lower()]

    bona = [r for r in indic_rows if r[1].strip().lower() == "bonafide"]
    spoof = [r for r in indic_rows if r[1].strip().lower() != "bonafide"]

    rng = random.Random(args.seed)
    rng.shuffle(bona)
    rng.shuffle(spoof)

    frac = args.indic_n / max(len(indic_rows), 1)
    n_bona = max(1, int(len(bona) * frac))
    n_spoof = max(1, args.indic_n - n_bona)

    kept = mlaad_rows + bona[:n_bona] + spoof[:n_spoof]
    rng.shuffle(kept)

    out_path = args.csv_path.rsplit(".", 1)[0] + f".mlaadkept{args.indic_n}.csv"
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(kept)
    print(f"[write] {out_path}: {len(kept)} rows "
          f"(mlaad={len(mlaad_rows)} indic_bona={n_bona} indic_spoof={n_spoof})")


if __name__ == "__main__":
    main()