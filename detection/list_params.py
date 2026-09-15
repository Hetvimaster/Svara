import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), "aasist_repo"))
from models.AASIST import Model as AASISTModel

with open("aasist_repo/config/AASIST-L.conf") as f:
    config = json.load(f)

model = AASISTModel(config["model_config"])
for name, p in model.named_parameters():
    print(name, tuple(p.shape))