import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import os
import sys
from pathlib import Path

# add repo root so util.py can be imported even when cwd differs
REPO_ROOT = Path(__file__).resolve().parents[1]  # adjust parents if nested deeper
sys.path.append(str(REPO_ROOT))
from util import save_labels

from resnn import ResidualMLPClassifier

def load_test_inputs(path):
    data = np.loadtxt(path, delimiter=',', dtype=np.float32)
    if data.ndim == 1:
        data = np.expand_dims(data, 0)
    return torch.from_numpy(data)

def infer(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for (inputs,) in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(preds)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualMLPClassifier(input_dim=512, n_classes=100, num_blocks=12, hidden_dim=4096, dropout=0.1)
    model.load_state_dict(torch.load("./resnn_checkpoints/resnn-best.pth", map_location=device))
    model.to(device)

    inputs = load_test_inputs("../../data/test.csv")
    loader = DataLoader(TensorDataset(inputs), batch_size=256, shuffle=False)
    predictions = infer(model, loader, device)

    output_path = Path("./resnn_checkpoints/predictions.csv")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    save_labels(str(output_path), predictions.numpy())