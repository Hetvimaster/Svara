"""
Fine-tune the ECAPA-TDNN embedding model on Indian-language speaker data.

Expected input CSV, no header:
    filepath,speaker_id
    /data/indic_speakers/spk001/utt1.wav,spk001
    /data/indic_speakers/spk001/utt2.wav,spk001
    /data/indic_speakers/spk002/utt1.wav,spk002
    ...
Needs several utterances per speaker (5-10 minimum) — this is why IndicSynth's
per-speaker hour counts matter; SEA-Spoof is spoof-labeled, not speaker-labeled,
so it's NOT usable for this script (only for the AASIST script above).

Only the embedding_model weights are touched/saved — the classification head
here exists purely to shape the embedding space during training and is
discarded afterward, exactly like the original VoxCeleb training recipe.
"""

import os
import csv
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np

from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
from speechbrain.lobes.features import Fbank
from speechbrain.nnet.losses import LogSoftmaxWrapper, AdditiveAngularMargin

SAMPLE_RATE = 16000


class SpeakerCSVDataset(Dataset):
    def __init__(self, csv_path, spk2idx=None, max_len_sec=4.0):
        self.items = []
        speakers = set()
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) != 2:
                    continue
                filepath, spk = row
                self.items.append((filepath.strip(), spk.strip()))
                speakers.add(spk.strip())

        self.spk2idx = spk2idx or {s: i for i, s in enumerate(sorted(speakers))}
        self.max_len = int(max_len_sec * SAMPLE_RATE)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filepath, spk = self.items[idx]
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)

        if len(audio) >= self.max_len:
            start = np.random.randint(0, len(audio) - self.max_len + 1)
            audio = audio[start:start + self.max_len]
        else:
            reps = int(self.max_len / len(audio)) + 1
            audio = np.tile(audio, reps)[: self.max_len]

        return torch.tensor(audio, dtype=torch.float32), self.spk2idx[spk]


def collate(batch):
    wavs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return wavs, labels


def load_pretrained_ecapa(checkpoint_path, device):
    """Loads the same embedding_model.ckpt your speaker_verification.py already
    downloaded via speechbrain, but as a raw trainable nn.Module rather than
    through the inference-only EncoderClassifier wrapper."""
    model = ECAPA_TDNN(
        input_size=80,  # matches 80-dim Fbank features below
        channels=[512, 512, 512, 512, 1536],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        lin_neurons=192,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={len(missing)} unexpected={len(unexpected)} keys when loading — "
              f"double check speechbrain version matches the checkpoint's architecture.")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument(
        "--init_checkpoint",
        default="pretrained_models/spkrec-ecapa-voxceleb/embedding_model.ckpt",
    )
    ap.add_argument("--out_checkpoint", default="pretrained_models/embedding_model_indic_finetuned.ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")    
    print(f"[device] {device}")

    train_ds = SpeakerCSVDataset(args.train_csv)
    val_ds = SpeakerCSVDataset(args.val_csv, spk2idx=train_ds.spk2idx)
    n_speakers = len(train_ds.spk2idx)
    print(f"[data] {n_speakers} speakers, {len(train_ds)} train utts, {len(val_ds)} val utts")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=2)

    fbank = Fbank(n_mels=80).to(device)
    embedding_model = load_pretrained_ecapa(args.init_checkpoint, device)
    classifier_head = nn.Linear(192, n_speakers, bias=False).to(device)
    aam = AdditiveAngularMargin(margin=0.2, scale=30).to(device)
    criterion = LogSoftmaxWrapper(loss_fn=aam)

    params = list(embedding_model.parameters()) + list(classifier_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        embedding_model.train()
        classifier_head.train()
        running_loss, n = 0.0, 0
        for wavs, labels in train_loader:
            wavs, labels = wavs.to(device), labels.to(device)
            feats = fbank(wavs)                      # (B, T, 80)
            emb = embedding_model(feats).squeeze(1)   # (B, 192)
            logits = classifier_head(emb)             # (B, n_speakers)
            loss = criterion(logits.unsqueeze(1), labels.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * wavs.size(0)
            n += wavs.size(0)
        train_loss = running_loss / n

        embedding_model.eval()
        classifier_head.eval()
        val_loss, vn = 0.0, 0
        with torch.no_grad():
            for wavs, labels in val_loader:
                wavs, labels = wavs.to(device), labels.to(device)
                feats = fbank(wavs)
                emb = embedding_model(feats).squeeze(1)
                logits = classifier_head(emb)
                loss = criterion(logits.unsqueeze(1), labels.unsqueeze(1))
                val_loss += loss.item() * wavs.size(0)
                vn += wavs.size(0)
        val_loss /= vn
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(embedding_model.state_dict(), args.out_checkpoint)
            print(f"  -> saved new best embedding checkpoint to {args.out_checkpoint}")

    print(f"[done] best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()