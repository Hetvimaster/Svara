"""
Evaluate a checkpoint's EER on a val CSV, no training. Use this to sanity-check
that a fine-tune actually improved things vs. the base checkpoint, on the
exact same val set.
"""
import argparse, json, sys, os
import torch
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "aasist_repo"))
from models.AASIST import Model as AASISTModel  
from finetune_aasist import SpoofCSVDataset, compute_eer  
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="aasist_repo/config/AASIST-L.conf")
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    with open(args.config) as f:
        config = json.load(f)
    model = AASISTModel(config["model_config"]).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    val_ds = SpoofCSVDataset(args.val_csv, train=False)
    n_bona = sum(1 for _, y in val_ds.items if y == 1)
    n_spoof = sum(1 for _, y in val_ds.items if y == 0)
    print(f"[val set] bonafide={n_bona} spoof={n_spoof}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    scores, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, logits = model(x)
            scores.extend(logits[:, 0].cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    eer = compute_eer(np.array(scores), np.array(labels))
    print(f"[{args.checkpoint}] val_EER={eer*100:.2f}%")


if __name__ == "__main__":
    main()