import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

from .data.dataset_lstm import LSTMMultiDataset
from .model.lstm_policy import LSTMMultiPolicy

def load_model(path, num_actions, device):
    model = LSTMMultiPolicy(num_actions=num_actions, pretrained=False).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def evaluate(model, loader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for seq_s, seq_l, seq_a in loader:
            seq_s, seq_l, seq_a = seq_s.to(device), seq_l.to(device), seq_a.to(device)
            probs = model(seq_s, seq_l)            # [1,T,6]
            preds = probs.argmax(dim=-1).cpu().numpy().ravel()
            trues = seq_a.cpu().numpy().ravel()
            y_pred.append(preds); y_true.append(trues)
    return np.concatenate(y_true), np.concatenate(y_pred)

def plot_cm(cm, classes):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i,j],
                     ha="center",
                     color="white" if cm[i,j]>thresh else "black")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_root    = "imitation_data_val"
    num_actions = 6

    ds = LSTMMultiDataset(val_root)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4)

    model = load_model("best_lstm_multi.pth", num_actions, device)
    y_true, y_pred = evaluate(model, loader, device)

    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Acc: {acc*100:.2f}%\n")

    names = [f"class_{i}" for i in range(num_actions)]
    print("Per-class report:\n")
    print(classification_report(y_true, y_pred, target_names=names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_cm(cm, names)
