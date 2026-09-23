"""
Download a small, targeted subset of MattyB95/VoxCelebSpoof.

Fixes the timeout in the old version: instead of streaming audio out of
remote zips one file at a time (many small HTTP range requests -> constant
timeouts), this pulls a couple of whole zip shards with hf_hub_download
(resumable, proper retries — same pattern download_mlaad_subset.py already
uses successfully), extracts a capped number of files locally, then deletes
the zip.

Repo layout (inferred from your error log paths):
    Chunk/<N>.zip            -> bonafide
    Clone/<model>/<N>.zip    -> spoof
If this guess is wrong, run:
    python -c "from huggingface_hub import HfApi; print([e.path for e in HfApi().list_repo_tree('MattyB95/VoxCelebSpoof', repo_type='dataset')])"
and adjust the two `list_dir(...)` calls below.
"""
import argparse, csv, os, random, shutil, zipfile
from huggingface_hub import HfApi, hf_hub_download

REPO = "MattyB95/VoxCelebSpoof"
AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a")


def list_dir(api, path):
    return list(api.list_repo_tree(REPO, path_in_repo=path, repo_type="dataset", recursive=False))


def pick_zips(api, path, n, rng):
    entries = [e for e in list_dir(api, path) if e.path.endswith(".zip")]
    rng.shuffle(entries)
    return entries[:n]


def extract_capped(zip_path, label, start_idx, cap, out_dir):
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(AUDIO_EXTS)]
        random.Random(0).shuffle(names)
        for name in names:
            if len(written) >= cap:
                break
            ext = name.rsplit(".", 1)[-1]
            out_name = f"{label}_{start_idx + len(written):04d}.{ext}"
            out_path = os.path.abspath(os.path.join(out_dir, out_name))
            with zf.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append((out_path, label))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/voxcelebspoof_cache")
    ap.add_argument("--out_train", default="data/voxcelebspoof_train.csv")
    ap.add_argument("--out_val", default="data/voxcelebspoof_val.csv")
    ap.add_argument("--max_per_class", type=int, default=150)
    ap.add_argument("--zips_per_class", type=int, default=2,
                     help="how many zip shards to pull per class — keep low, shards can be large")
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--keep_zips", action="store_true",
                     help="don't delete downloaded zips after extracting (default: delete, saves disk)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp_dir = os.path.join(args.out_dir, "_zips")
    os.makedirs(tmp_dir, exist_ok=True)
    rng = random.Random(args.seed)
    api = HfApi()

    bona_zips = pick_zips(api, "Chunk", args.zips_per_class, rng)
    clone_models = [e.path for e in list_dir(api, "Clone")]
    rng.shuffle(clone_models)
    spoof_zips = []
    for model_dir in clone_models:
        if len(spoof_zips) >= args.zips_per_class:
            break
        spoof_zips.extend(pick_zips(api, model_dir, 1, rng))

    bona_rows, spoof_rows = [], []
    for label, zips, bucket in [("bonafide", bona_zips, bona_rows), ("spoof", spoof_zips, spoof_rows)]:
        per_zip_cap = max(1, args.max_per_class // max(len(zips), 1))
        for entry in zips:
            print(f"[download] {entry.path} ...")
            local_zip = hf_hub_download(REPO, filename=entry.path, repo_type="dataset", local_dir=tmp_dir)
            bucket.extend(extract_capped(local_zip, label, len(bucket), per_zip_cap, args.out_dir))
            if not args.keep_zips:
                os.remove(local_zip)
            print(f"  [{label}] have {len(bucket)}/{args.max_per_class}")
            if len(bucket) >= args.max_per_class:
                break

    print(f"[done] bonafide={len(bona_rows)} spoof={len(spoof_rows)}")
    if not bona_rows or not spoof_rows:
        raise SystemExit("[error] one class came back empty — check the repo layout with list_repo_tree")

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