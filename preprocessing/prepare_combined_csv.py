"""
Pulls a capped, balanced subset from BOTH IndicSynth (spoof) and Kathbath
(bonafide) across all 12 shared Indian languages, and writes ONE combined
train/val CSV pair for finetune_aasist.py.

Requires: pip install datasets huggingface_hub soundfile
Requires: huggingface-cli login (Kathbath is gated — accept terms on its HF page first)
Requires: ffmpeg on PATH (Kathbath ships .m4a; soundfile can't read that directly)
"""

import os
import csv
import random
import subprocess
import argparse

from datasets import load_dataset
import soundfile as sf

# exact 12-language overlap between IndicSynth and Kathbath
LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
    "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
]


def save_wav(audio, sr, out_path):
    sf.write(out_path, audio, sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/combined")
    ap.add_argument("--per_language_cap", type=int, default=150,
                     help="clips per class per language — keep this small on an M1 Air; "
                          "150 spoof + 150 bonafide x 12 languages = ~3,600 clips total")
    ap.add_argument("--val_fraction", type=float, default=0.15)
    args = ap.parse_args()

    train_rows, val_rows = [], []

    for lang in LANGUAGES:
        lang_dir = os.path.join(args.out_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)

        # --- spoof side: IndicSynth ---
        spoof_ds = load_dataset("vdivyasharma/IndicSynth", name=lang, split="train", streaming=True)
        spoof_rows = []
        for i, ex in enumerate(spoof_ds):
            if i >= args.per_language_cap:
                break
            fpath = os.path.abspath(os.path.join(lang_dir, f"spoof_{i:04d}.wav"))
            save_wav(ex["audio"]["array"], ex["audio"]["sampling_rate"], fpath)
            spoof_rows.append((fpath, "spoof"))

                # --- bonafide side: Kathbath (new parquet repo, capital K) ---
        bona_ds = load_dataset("ai4bharat/Kathbath", lang.lower(), split="valid", streaming=True)
        bona_rows = []
        for i, ex in enumerate(bona_ds):
            if i >= args.per_language_cap:
                break
            fpath = os.path.abspath(os.path.join(lang_dir, f"bona_{i:04d}.wav"))
            save_wav(ex["audio_filepath"]["array"], ex["audio_filepath"]["sampling_rate"], fpath)
            bona_rows.append((fpath, "bonafide"))
        print(f"[{lang}] spoof={len(spoof_rows)} bonafide={len(bona_rows)}")

        # per-language train/val split, so val isn't accidentally missing a language
        combined = spoof_rows + bona_rows
        random.shuffle(combined)
        cut = int(len(combined) * (1 - args.val_fraction))
        train_rows.extend(combined[:cut])
        val_rows.extend(combined[cut:])

    random.shuffle(train_rows)
    random.shuffle(val_rows)

    for rows, name in [(train_rows, "train.csv"), (val_rows, "val.csv")]:
        with open(os.path.join(args.out_dir, name), "w", newline="") as f:
            csv.writer(f).writerows(rows)
        print(f"[done] wrote {len(rows)} rows to {name}")


if __name__ == "__main__":
    main()