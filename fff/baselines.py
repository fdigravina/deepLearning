import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, n_layers=1):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MLPClassifier(MLP):
    def __init__(self, in_dim, n_classes, hidden, n_layers=1):
        super().__init__(in_dim, n_classes, hidden, n_layers)