"""
Try to decode every file listed in one or more filepath,label CSVs with the
exact same soundfile call finetune_aasist.py uses. Reports which files fail
and why, and can write a cleaned copy with bad rows dropped.
"""
import argparse, csv, sys
import soundfile as sf


def check_one(path):
    try:
        audio, sr = sf.read(path)
        if audio.size == 0:
            return False, "empty audio (0 samples)"
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="one or more filepath,label CSVs")
    ap.add_argument("--write_clean", action="store_true",
                    help="write <name>.clean.csv next to each input, bad rows dropped")
    args = ap.parse_args()

    total_bad = 0
    for csv_path in args.csvs:
        rows = []
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))

        bad_rows = []
        good_rows = []
        for i, row in enumerate(rows):
            if len(row) != 2:
                bad_rows.append((row, "malformed row (not 2 columns)"))
                continue
            fp, label = row
            ok, reason = check_one(fp.strip())
            if ok:
                good_rows.append(row)
            else:
                bad_rows.append((row, reason))
            if (i + 1) % 200 == 0:
                print(f"  ...checked {i+1}/{len(rows)}")

        print(f"\n[{csv_path}] {len(good_rows)} ok, {len(bad_rows)} bad")
        for row, reason in bad_rows[:30]:
            print(f"  BAD: {row} -> {reason}")
        if len(bad_rows) > 30:
            print(f"  ... and {len(bad_rows) - 30} more")

        total_bad += len(bad_rows)
        if args.write_clean and bad_rows:
            out_path = csv_path.rsplit(".", 1)[0] + ".clean.csv"
            with open(out_path, "w", newline="") as f:
                csv.writer(f).writerows(good_rows)
            print(f"  -> wrote {out_path} ({len(good_rows)} rows)")

    sys.exit(1 if total_bad else 0)


if __name__ == "__main__":
    main()