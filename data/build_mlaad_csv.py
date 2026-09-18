"""
Build filepath,label CSVs from a SUBSET of MLAAD.

MLAAD layout:
    <root>/fake/<language>/<model_name>/meta.csv
    <root>/fake/<language>/<model_name>/*.wav

meta.csv is PIPE-delimited with columns:
    path|original_file|language|is_original_language|duration|
    training_data|model_name|architecture|transcript|reference_speaker

MLAAD is spoof-only. Bonafide must come from M-AILABS (--mailabs_root) or
from your existing data/combined CSVs at merge time. See --help.

Split is MODEL-disjoint, not clip-disjoint: val holds out entire TTS systems
so val EER actually measures generalization to unseen generators, which is
the exact failure you're debugging.
"""
import argparse, csv, glob, os, random

META_DELIM = "|"


def load_meta(meta_path):
    """Yield (abs_wav_path, model_name, original_file) per row."""
    model_dir = os.path.dirname(meta_path)
    mlaad_root = os.path.abspath(os.path.join(model_dir, "..", "..", ".."))
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter=META_DELIM):
            rel = row.get("path", "").strip()
            if not rel:
                continue
            # 'path' may be repo-relative ("fake/en/model/x.wav") or a bare
            # filename depending on version — try both, keep what exists.
            cand = [
                os.path.join(model_dir, os.path.basename(rel)),
                os.path.join(mlaad_root, rel),
            ]
            wav = next((p for p in cand if os.path.isfile(p)), None)
            if wav is None:
                continue
            yield wav, row.get("model_name", "unknown"), row.get("original_file", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="path to your MLAAD copy")
    ap.add_argument("--languages", nargs="+", required=True,
                    help="language subdirs under fake/, e.g. en de hi")
    ap.add_argument("--max_models_per_language", type=int, default=6,
                    help="cap number of TTS systems kept per language")
    ap.add_argument("--max_clips_per_model", type=int, default=60,
                    help="cap clips per TTS system — this is your subset knob")
    ap.add_argument("--mailabs_root", default=None,
                    help="optional: M-AILABS root, to add matched bonafide rows")
    ap.add_argument("--out_train", default="data/mlaad_train.csv")
    ap.add_argument("--out_val", default="data/mlaad_val.csv")
    ap.add_argument("--val_fraction_models", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows_by_model = {}   # model_name -> list of (path, label)

    for lang in args.languages:
        lang_dir = os.path.join(args.root, "fake", lang)
        metas = sorted(glob.glob(os.path.join(lang_dir, "*", "meta.csv")))
        if not metas:
            print(f"[warn] no meta.csv found under {lang_dir} — skipping '{lang}'")
            continue
        rng.shuffle(metas)
        metas = metas[:args.max_models_per_language]

        for meta_path in metas:
            entries = list(load_meta(meta_path))
            if not entries:
                continue
            rng.shuffle(entries)
            entries = entries[:args.max_clips_per_model]
            model = entries[0][1] or os.path.basename(os.path.dirname(meta_path))
            bucket = rows_by_model.setdefault(model, [])

            for wav, _, original_file in entries:
                bucket.append((wav, "spoof"))
                # matched bonafide from M-AILABS, if available
                if args.mailabs_root and original_file:
                    real = os.path.join(args.mailabs_root, original_file.strip())
                    if os.path.isfile(real):
                        bucket.append((real, "bonafide"))

    if not rows_by_model:
        raise SystemExit("[error] nothing collected — check --root and --languages")

    models = sorted(rows_by_model)
    rng.shuffle(models)
    n_val = max(1, int(len(models) * args.val_fraction_models))
    val_models = set(models[:n_val])

    train_rows, val_rows = [], []
    for m in models:
        (val_rows if m in val_models else train_rows).extend(rows_by_model[m])

    for rows, path in [(train_rows, args.out_train), (val_rows, args.out_val)]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(rows)
        n_bona = sum(1 for _, l in rows if l == "bonafide")
        print(f"[write] {path}: {len(rows)} rows "
              f"(spoof={len(rows)-n_bona} bonafide={n_bona})")
    print(f"[models] train={len(models)-n_val} val={n_val} (model-disjoint)")
    if not args.mailabs_root:
        print("[note] spoof-only output — make sure your merged train CSV has "
              "enough bonafide from data/combined, or pass --mailabs_root")


if __name__ == "__main__":
    main()