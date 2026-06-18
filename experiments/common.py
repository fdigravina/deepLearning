import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fff.mrfff import MRFFFClassifier

def get_dataset(name, seed=0):
    mapping = {
        "spambase": 44,
        "usps": 41082,
        "letter": 6,
        "satimage": 182,
    }
    if name not in mapping:
        raise ValueError(name)
    d = fetch_openml(data_id=mapping[name], as_frame=False, parser='auto')
    X = d.data
    y = d.target
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)
    X = X.astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y)
    sc = StandardScaler().fit(X_tr)
    X_tr = sc.transform(X_tr).astype(np.float32)
    X_te = sc.transform(X_te).astype(np.float32)
    to_t = lambda a, t: torch.tensor(a, dtype=t)
    return (to_t(X_tr, torch.float32), to_t(y_tr, torch.long),
            to_t(X_te, torch.float32), to_t(y_te, torch.long),
            X.shape[1], int(y.max()) + 1)

def train_model(model, X_tr, y_tr, epochs=60, lr=1e-2, weight_decay=1e-4,
                batch_size=256, reg_arity=0.0, reg_split=0.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    n = X_tr.shape[0]
    is_mr = isinstance(model, MRFFFClassifier)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            if is_mr:
                loss = model.loss(X_tr[idx], y_tr[idx],
                                  reg_split=reg_split, reg_arity=reg_arity)
            else:
                out = model(X_tr[idx])
                loss = loss_fn(out, y_tr[idx])
            loss.backward()
            opt.step()
    return model

@torch.no_grad()
def evaluate(model, X_te, y_te):
    model.eval()
    pred = model(X_te).argmax(-1)
    return (pred == y_te).float().mean().item()