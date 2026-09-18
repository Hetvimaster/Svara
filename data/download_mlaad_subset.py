"""
Download a small, targeted slice of MLAAD without enumerating the whole repo.

Walks fake/<lang>/ one level at a time (cheap), picks N models, then pulls
meta.csv + the first K wavs from each. Prints progress so you can see it work.
"""
import argparse, os, random
from huggingface_hub import HfApi, hf_hub_download

REPO = "mueller91/MLAAD"


def list_dir(api, path):
    """Non-recursive listing of one directory in the dataset repo."""
    return list(api.list_repo_tree(REPO, path_in_repo=path,
                                   repo_type="dataset", recursive=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--languages", nargs="+", default=["en"])
    ap.add_argument("--models_per_language", type=int, default=4)
    ap.add_argument("--clips_per_model", type=int, default=50)
    ap.add_argument("--local_dir", default="data/mlaad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--list_only", action="store_true",
                    help="just print available languages and exit")
    args = ap.parse_args()

    api = HfApi()
    rng = random.Random(args.seed)

    if args.list_only:
        print("[languages available under fake/]")
        for e in list_dir(api, "fake"):
            print("  ", os.path.basename(e.path))
        return

    for lang in args.languages:
        lang_path = f"fake/{lang}"
        try:
            model_dirs = [e.path for e in list_dir(api, lang_path)
                          if getattr(e, "tree_id", None) or not e.path.endswith((".wav", ".csv"))]
        except Exception as exc:
            print(f"[warn] cannot list {lang_path}: {exc}")
            continue

        if not model_dirs:
            print(f"[warn] no model folders under {lang_path}")
            continue

        rng.shuffle(model_dirs)
        model_dirs = model_dirs[:args.models_per_language]
        print(f"[{lang}] {len(model_dirs)} models selected")

        for mdir in model_dirs:
            entries = list_dir(api, mdir)
            wavs = sorted(e.path for e in entries if e.path.endswith(".wav"))
            metas = [e.path for e in entries if e.path.endswith("meta.csv")]
            wavs = wavs[:args.clips_per_model]
            targets = metas + wavs

            print(f"  {os.path.basename(mdir)}: {len(wavs)} wavs + "
                  f"{len(metas)} meta")
            for i, f in enumerate(targets, 1):
                hf_hub_download(REPO, filename=f, repo_type="dataset",
                                local_dir=args.local_dir)
                if i % 25 == 0:
                    print(f"    ... {i}/{len(targets)}")

    print(f"[done] files under {args.local_dir}/fake/")


if __name__ == "__main__":
    main()