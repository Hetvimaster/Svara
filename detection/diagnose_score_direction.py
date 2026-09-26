"""
Check whether logits[:,1] is actually scoring in the direction
finetune_aasist.py/eval_checkpoint.py assume (higher = more bonafide-like).
Prints the mean score per class -- if bonafide's mean is LOWER than spoof's,
the convention is inverted and EER numbers everywhere need to be recomputed
with the sign flipped.
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
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    scores1, scores0, labels = [], [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            _, logits = model(x)
            scores1.extend(logits[:, 1].cpu().numpy().tolist())  # current assumption: bonafide score
            scores0.extend(logits[:, 0].cpu().numpy().tolist())  # the other head
            labels.extend(y.numpy().tolist())

    scores1, scores0, labels = np.array(scores1), np.array(scores0), np.array(labels)
    bona_mask = labels == 1
    spoof_mask = labels == 0

    print(f"[logits[:,1]] mean(bonafide)={scores1[bona_mask].mean():.4f} "
          f"mean(spoof)={scores1[spoof_mask].mean():.4f}")
    print(f"[logits[:,0]] mean(bonafide)={scores0[bona_mask].mean():.4f} "
          f"mean(spoof)={scores0[spoof_mask].mean():.4f}")

    eer_as_is = compute_eer(scores1, labels)
    eer_flipped = compute_eer(-scores1, labels)
    eer_other_head = compute_eer(scores0, labels)
    print(f"\nEER using logits[:,1] as-is:      {eer_as_is*100:.2f}%")
    print(f"EER using -logits[:,1] (flipped): {eer_flipped*100:.2f}%")
    print(f"EER using logits[:,0] as-is:      {eer_other_head*100:.2f}%")
    print("\nWhichever of the three is lowest tells you the correct scoring convention.")


if __name__ == "__main__":
    main()