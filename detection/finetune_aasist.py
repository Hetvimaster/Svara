"""
Fine-tune AASIST-L on Indian-language spoof data (IndicSynth / SEA-Spoof).

Expected input: a CSV with two columns, no header:
    filepath,label
    /data/indicsynth/hi/utt001.wav,bonafide
    /data/indicsynth/hi/utt002.wav,spoof
    ...
Convert whatever format IndicSynth/SEA-Spoof ship in into this CSV first —
that's a data-wrangling step, not a modeling one, do it separately per dataset
and just concatenate the CSVs (this is also where you scope down to your
2-3 demo languages, per your own feasibility table).

Label convention matches the ORIGINAL AASIST training code exactly:
    bonafide -> 1, spoof -> 0
Do not flip this — the pretrained checkpoint's output head already expects it.
"""

import sys
import os
import json
import csv
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import numpy as np

# make the cloned repo importable — run this script from detection/
sys.path.append(os.path.join(os.path.dirname(__file__), "aasist_repo"))
from models.AASIST import Model as AASISTModel  # noqa: E402

SAMPLE_RATE = 16000
NB_SAMP = 64600  # ~4s at 16kHz, fixed by AASIST-L architecture


def pad_random(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Random crop for training (matches original data_utils.py behavior —
    keeps the model from memorizing fixed start-of-clip artifacts)."""
    x_len = x.shape[0]
    if x_len >= max_len:
        start = np.random.randint(0, x_len - max_len + 1)
        return x[start:start + max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def pad_fixed(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Deterministic pad/trim for validation — reproducible scores."""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


def trim_silence(audio: np.ndarray, sr: int, top_db: float = 30) -> np.ndarray:
    """Trim leading/trailing near-silence. Without this, real recordings
    (which often have room tone / mic handling noise at the edges) and
    clean TTS clips (which usually don't) differ systematically in a way
    that has nothing to do with spoof artifacts — the model can learn
    that shortcut instead of the real one."""
    if len(audio) == 0:
        return audio
    frame = max(int(sr * 0.02), 1)
    n_frames = max(len(audio) // frame, 1)
    energy = np.array([
        np.sqrt(np.mean(audio[i * frame:(i + 1) * frame] ** 2) + 1e-12)
        for i in range(n_frames)
    ])
    ref = energy.max() if energy.max() > 0 else 1.0
    db = 20 * np.log10(energy / ref + 1e-12)
    voiced = np.where(db > -top_db)[0]
    if len(voiced) == 0:
        return audio
    start = voiced[0] * frame
    end = min((voiced[-1] + 1) * frame, len(audio))
    trimmed = audio[start:end]
    return trimmed if len(trimmed) > 0 else audio


def apply_call_channel(audio: np.ndarray, sr: int, rng: random.Random) -> np.ndarray:
    """Randomly simulate voice-call-style degradation during training:
    telephone bandpass (300-3400 Hz), an 8kHz codec-like round trip, and a
    brief dropout. Applied probabilistically so the model sees BOTH clean
    and call-degraded audio for both classes, rather than the two
    conditions being confounded with the two labels."""
    import librosa
    from scipy.signal import butter, sosfilt

    out = audio.copy()

    if rng.random() < 0.5:
        sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
        out = sosfilt(sos, out).astype(np.float32)

    if rng.random() < 0.5:
        narrow = librosa.resample(out.astype(np.float32), orig_sr=sr, target_sr=8000)
        out = librosa.resample(narrow, orig_sr=8000, target_sr=sr).astype(np.float32)

    if rng.random() < 0.3 and len(out) > sr // 2:
        drop_len = rng.randint(int(sr * 0.05), int(sr * 0.15))
        drop_start = rng.randint(0, len(out) - drop_len)
        out[drop_start:drop_start + drop_len] = 0.0

    return out


class SpoofCSVDataset(Dataset):
    def __init__(self, csv_path, train=True, augment_channel=False, augment_seed=0):
        self.items = []
        with open(csv_path, newline="") as f:
            for row in csv.reader(f):
                if len(row) != 2:
                    continue
                filepath, label = row
                y = 1 if label.strip().lower() == "bonafide" else 0
                self.items.append((filepath.strip(), y))
        self.train = train
        self.augment_channel = augment_channel
        self._rng = random.Random(augment_seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        filepath, y = self.items[idx]
        audio, sr = sf.read(filepath)
        if audio.ndim > 1:  # collapse stereo -> mono
            audio = audio.mean(axis=1)
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = audio.astype(np.float32)
        audio = trim_silence(audio, SAMPLE_RATE)  # both train and val — keep it consistent
        if self.train and self.augment_channel:
            audio = apply_call_channel(audio, SAMPLE_RATE, self._rng)
        audio = pad_random(audio) if self.train else pad_fixed(audio)
        return torch.tensor(audio, dtype=torch.float32), y


def class_weights_from_dataset(dataset: SpoofCSVDataset, device):
    """Original AASIST used a fixed [0.1, 0.9] weight tuned for ASVspoof2019 LA's
    class imbalance (~10:1 spoof:bonafide). IndicSynth/SEA-Spoof will have a
    different ratio, so compute it from your actual fine-tuning data instead
    of copying that constant."""
    n_spoof = sum(1 for _, y in dataset.items if y == 0)
    n_bona = sum(1 for _, y in dataset.items if y == 1)
    total = n_spoof + n_bona
    # inverse-frequency weighting, normalized
    w_spoof = total / (2 * max(n_spoof, 1))
    w_bona = total / (2 * max(n_bona, 1))
    print(f"[data] bonafide={n_bona} spoof={n_spoof} -> weights spoof={w_spoof:.3f} bona={w_bona:.3f}")
    return torch.tensor([w_spoof, w_bona], dtype=torch.float32).to(device)


def compute_eer(scores, labels):
    """scores: higher = more bonafide-like (matches batch_out[:,1] convention).
    labels: 1=bonafide, 0=spoof. Requires scikit-learn (pip install scikit-learn)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[idx] + fnr[idx]) / 2


def freeze_backbone_except_head(model, unfreeze_prefixes=(
        "out_layer", "GAT_layer_S", "GAT_layer_T",
        "HtrgGAT_layer_ST11", "HtrgGAT_layer_ST12",
        "HtrgGAT_layer_ST21", "HtrgGAT_layer_ST22",
        "pool_S", "pool_T", "pool_hS1", "pool_hT1", "pool_hS2", "pool_hT2")):
    """Freeze the graph-attention backbone, fine-tune only the output head and
    the last heterogeneous-GAT blocks. Name-based rather than index-based —
    the old index heuristic unfroze only tiny trailing bias tensors (~0.5% of
    the model), which meant training did essentially nothing."""
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(prefix) for prefix in unfreeze_prefixes)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[freeze] trainable params: {trainable}/{total_params} "
          f"({100*trainable/total_params:.1f}%)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--config", default="aasist_repo/config/AASIST-L.conf")
    ap.add_argument("--init_checkpoint", default="aasist_repo/models/weights/AASIST-L.pth")
    ap.add_argument("--out_checkpoint", default="aasist_repo/models/weights/AASIST-L_finetuned_indic.pth")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3,
                    help="stop early if val EER doesn't improve for this many epochs")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)  # small — fine-tuning, not training from scratch
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--augment_channel", action="store_true",help="randomly apply telephone bandpass / 8kHz round-trip / dropout to TRAIN audio only")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")    
    print(f"[device] {device}")

    with open(args.config) as f:
        config = json.load(f)

    model = AASISTModel(config["model_config"]).to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
    print(f"[model] loaded pretrained weights from {args.init_checkpoint}")

    if args.freeze_backbone:
        freeze_backbone_except_head(model)
        
    train_ds = SpoofCSVDataset(args.train_csv, train=True, augment_channel=args.augment_channel)
    val_ds = SpoofCSVDataset(args.val_csv, train=False)  # val stays clean — augmentation is a train-time trick only
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    weight = class_weights_from_dataset(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=1e-4)
    def set_frozen_bn_eval(model):
        """model.train() re-enables BatchNorm running-stat updates on every
        submodule regardless of requires_grad. If a module's parameters are
        all frozen, force it back to eval() so its stats stop drifting."""
        for module in model.modules():
            params = list(module.parameters(recurse=False))
            if params and all(not p.requires_grad for p in params):
                module.eval()

    best_eer = 1.0
    epochs_no_improve = 0
    for epoch in range(args.epochs):
        model.train()
        if args.freeze_backbone:
            set_frozen_bn_eval(model)
        running_loss, n = 0.0, 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            _, logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            if batch_idx % 20 == 0:
                print(f"  [batch {batch_idx}/{len(train_loader)}] loss={loss.item():.4f}")
        train_loss = running_loss / n

        model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                _, logits = model(x)
                score = logits[:, 0].cpu().numpy()  
                all_scores.extend(score.tolist())
                all_labels.extend(y.numpy().tolist())
        eer = compute_eer(np.array(all_scores), np.array(all_labels))
        print(f"[epoch {epoch}] train_loss={train_loss:.4f} val_EER={eer*100:.2f}%")
        if eer < best_eer:
            best_eer = eer
            epochs_no_improve = 0
            torch.save(model.state_dict(), args.out_checkpoint)
            print(f"  -> saved new best checkpoint (EER={eer*100:.2f}%) to {args.out_checkpoint}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"[early stop] no improvement for {args.patience} epochs, stopping")
                break

        if device == "mps":
            torch.mps.empty_cache()

    print(f"[done] best val EER: {best_eer*100:.2f}%")


if __name__ == "__main__":
    main()