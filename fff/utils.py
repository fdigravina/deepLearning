import math
import os
import random
import time
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def time_inference(model, x, repeats=20, warmup=3):
    model.eval()
    for _ in range(warmup):
        model(x)
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        best = min(best, time.perf_counter() - t0)
    return best / x.shape[0]


def mlp_flops(in_dim, hidden, out_dim, n_layers=1):
    flops = 2 * in_dim * hidden
    flops += 2 * hidden * hidden * (n_layers - 1)
    flops += 2 * hidden * out_dim
    return flops


def fff_flops(in_dim, out_dim, depth, leaf_width=0):
    router = 2 * depth * in_dim
    if leaf_width > 0:
        leaf = 2 * in_dim * leaf_width + 2 * leaf_width * out_dim
    else:
        leaf = 2 * in_dim * out_dim
    return router + leaf


class Logger:
    def __init__(self, path, echo=True):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.echo = echo
        self._fh = open(path, "w")

    def __call__(self, line=""):
        self._fh.write(line + "\n")
        self._fh.flush()
        if self.echo:
            print(line)

    def close(self):
        self._fh.close()