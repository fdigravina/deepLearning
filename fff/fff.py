import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FastFeedforward(nn.Module):
    def __init__(self, in_dim, out_dim, depth, leaf_width=0, region_leak=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.depth = depth
        self.n_nodes = 2 ** depth - 1
        self.n_leaves = 2 ** depth
        self.leaf_width = leaf_width
        self.region_leak = region_leak

        self.node_w = nn.Parameter(torch.empty(self.n_nodes, in_dim))
        self.node_b = nn.Parameter(torch.zeros(self.n_nodes))

        if leaf_width > 0:
            self.l1_w = nn.Parameter(torch.empty(self.n_leaves, leaf_width, in_dim))
            self.l1_b = nn.Parameter(torch.zeros(self.n_leaves, leaf_width))
            self.l2_w = nn.Parameter(torch.empty(self.n_leaves, out_dim, leaf_width))
            self.l2_b = nn.Parameter(torch.zeros(self.n_leaves, out_dim))
        else:
            self.l1_w = nn.Parameter(torch.empty(self.n_leaves, out_dim, in_dim))
            self.l1_b = nn.Parameter(torch.zeros(self.n_leaves, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.node_w, std=1.0 / math.sqrt(self.in_dim))
        bound = 1.0 / math.sqrt(self.in_dim)
        nn.init.uniform_(self.l1_w, -bound, bound)
        if self.leaf_width > 0:
            nn.init.uniform_(self.l2_w, -1.0 / math.sqrt(self.leaf_width),
                             1.0 / math.sqrt(self.leaf_width))

    def _leaf_forward(self, x, leaf_idx):
        w1 = self.l1_w[leaf_idx]
        b1 = self.l1_b[leaf_idx]
        h = torch.bmm(w1, x.unsqueeze(-1)).squeeze(-1) + b1
        if self.leaf_width > 0:
            h = F.relu(h)
            w2 = self.l2_w[leaf_idx]
            b2 = self.l2_b[leaf_idx]
            h = torch.bmm(w2, h.unsqueeze(-1)).squeeze(-1) + b2
        return h

    def _all_leaves_forward(self, x):
        h = torch.einsum("loi,bi->blo", self.l1_w, x) + self.l1_b
        if self.leaf_width > 0:
            h = F.relu(h)
            h = torch.einsum("loh,blh->blo", self.l2_w, h) + self.l2_b
        return h

    def forward(self, x):
        if not self.training:
            return self._hard_forward(x)
        return self._soft_forward(x)

    def _soft_forward(self, x):
        logits = F.linear(x, self.node_w, self.node_b)
        c = torch.sigmoid(logits)
        if self.region_leak > 0.0:
            c = c * (1 - self.region_leak) + 0.5 * self.region_leak

        prob = x.new_ones(x.shape[0], 1)
        offset = 0
        for level in range(self.depth):
            n_level = 2 ** level
            c_level = c[:, offset:offset + n_level]
            left = prob * (1 - c_level)
            right = prob * c_level
            prob = torch.stack([left, right], dim=-1).reshape(x.shape[0], 2 * n_level)
            offset += n_level

        leaf_out = self._all_leaves_forward(x)
        return torch.einsum("bl,blo->bo", prob, leaf_out)

    @torch.no_grad()
    def _hard_forward(self, x):
        pos = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        for level in range(self.depth):
            node_flat = (2 ** level - 1) + pos
            w = self.node_w[node_flat]
            b = self.node_b[node_flat]
            go_right = ((x * w).sum(-1) + b) > 0
            pos = pos * 2 + go_right.long()
        return self._leaf_forward(x, pos)


class FFFClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, depth, leaf_width=0, region_leak=0.0):
        super().__init__()
        self.fff = FastFeedforward(in_dim, n_classes, depth, leaf_width, region_leak)

    def forward(self, x):
        return self.fff(x)