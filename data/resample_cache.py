"""
One-time pass: read every file in a CSV, resample anything not already
16kHz mono to a cached copy, and rewrite the CSV to point at the cache.
Run once before training — saves a librosa.resample() call per file per epoch.
"""
import argparse, csv, os
import soundfile as sf
import numpy as np


def ensure_16k_mono(src_path, cache_dir, sample_rate=16000):
    src_path = os.path.abspath(src_path)
    cache_dir = os.path.abspath(cache_dir)
    info = sf.info(src_path)
    if info.samplerate == sample_rate and info.channels == 1:
        return src_path  # already fine, no copy needed

    os.makedirs(cache_dir, exist_ok=True)
    tag = "mlaad__" if "mlaad" in src_path.lower() else ""
    cache_path = os.path.join(cache_dir, tag + os.path.basename(src_path))
    if os.path.exists(cache_path):
        return cache_path

    audio, sr = sf.read(src_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
    sf.write(cache_path, audio.astype(np.float32), sample_rate)
    return cache_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--cache_dir", default="data/resampled_cache")
    args = ap.parse_args()

    for csv_path in args.csvs:
        rows = []
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))

        out_rows = []
        n_resampled = 0
        for i, (fp, label) in enumerate(rows):
            new_fp = ensure_16k_mono(fp.strip(), args.cache_dir)
            if new_fp != fp.strip():
                n_resampled += 1
            out_rows.append((new_fp, label))
            if (i + 1) % 200 == 0:
                print(f"  ...{i+1}/{len(rows)}")

        out_path = csv_path.rsplit(".", 1)[0] + ".cached.csv"
        with open(out_path, "w", newline="") as f:
            csv.writer(f).writerows(out_rows)
        print(f"[{csv_path}] resampled {n_resampled}/{len(rows)} -> {out_path}")


if __name__ == "__main__":
    main()