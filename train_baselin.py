import argparse
import numpy as np
import pandas as pd

# -------------------------------
# Daten laden & Sequenzen bauen
# -------------------------------
def make_sequences(df, win=30, step=1, feat_cols=("hip_deg","knee_deg","trunk_deg"), label_col="phase_id"):
    X, y = [], []
    vals = df[list(feat_cols)].to_numpy(dtype=np.float32)
    labels = df[label_col].to_numpy(dtype=np.int64)
    for start in range(0, len(df) - win, step):
        sl = slice(start, start+win)
        if (labels[sl] < 0).any():  # -1 = unbekannt
            continue
        X.append(vals[sl])
        # Mehrheitslabel im Fenster
        counts = np.bincount(labels[sl], minlength=3)
        y.append(np.argmax(counts))
    return np.array(X), np.array(y)

def load_data(csv_path, win=30, step=1):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["hip_deg","knee_deg","trunk_deg"])
    X, y = make_sequences(df, win=win, step=step)
    # einfache Normierung
    mu, std = X.mean(axis=(0,1), keepdims=True), X.std(axis=(0,1), keepdims=True)+1e-6
    X = (X - mu) / std
    # Split
    n = len(X)
    ntr = int(0.8 * n)
    return (X[:ntr], y[:ntr]), (X[ntr:], y[ntr:])

# -------------------------------
# PyTorch-Variante
# -------------------------------
def torch_train(csv_path, epochs=5, lr=1e-3, win=30, step=1):
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    (Xtr, ytr), (Xte, yte) = load_data(csv_path, win=win, step=step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    test_ds  = TensorDataset(torch.tensor(Xte), torch.tensor(yte))
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl  = DataLoader(test_ds, batch_size=128)

    class LSTMNet(nn.Module):
        def __init__(self, in_dim=3, hidden=64, num_layers=1, num_classes=3):
            super().__init__()
            self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, num_classes))
        def forward(self, x):
            out, _ = self.lstm(x)
            last = out[:,-1,:]
            return self.head(last)

    model = LSTMNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, epochs+1):
        model.train()
        tot = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item()*len(xb)
        model.eval()
        correct = 0
        n = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(1)
                correct += (preds==yb).sum().item()
                n += len(yb)
        print(f"[Torch] Epoch {ep}: train_loss={tot/len(train_ds):.4f} test_acc={correct/max(1,n):.3f}")

# -------------------------------
# TensorFlow-Variante
# -------------------------------
def tf_train(csv_path, epochs=5, lr=1e-3, win=30, step=1):
    import tensorflow as tf
    (Xtr, ytr), (Xte, yte) = load_data(csv_path, win=win, step=step)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xtr.shape[1], Xtr.shape[2])),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(3, activation="softmax")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(Xtr, ytr, validation_data=(Xte, yte), epochs=epochs, batch_size=64)
    print("[TF] Training fertig.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="runs/squat/features.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--framework", choices=["torch","tf"], default="torch")
    ap.add_argument("--win", type=int, default=30, help="Fensterlänge (Frames)")
    ap.add_argument("--step", type=int, default=1, help="Fensterschritt")
    args = ap.parse_args()

    if args.framework == "torch":
        torch_train(args.csv, epochs=args.epochs, win=args.win, step=args.step)
    else:
        tf_train(args.csv, epochs=args.epochs, win=args.win, step=args.step)
