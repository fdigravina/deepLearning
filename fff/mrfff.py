import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _heap_sizes(arity, depth):
    n_internal = (arity ** depth - 1) // (arity - 1)
    n_total = (arity ** (depth + 1) - 1) // (arity - 1)
    return n_internal, n_total

class MultiResolutionFFF(nn.Module):
    def __init__(self, in_dim, out_dim, max_depth, max_arity,
                 split_init=-1.0, gate_init=None, halt_threshold=0.5):
        super().__init__()
        if max_arity < 2:
            raise ValueError("max_arity must be >= 2")
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.max_depth = max_depth
        self.K = max_arity
        self.halt_threshold = halt_threshold
        self.n_internal, self.n_total = _heap_sizes(max_arity, max_depth)

        self.router_w = nn.Parameter(torch.empty(self.n_internal, max_arity, in_dim))
        self.router_b = nn.Parameter(torch.zeros(self.n_internal, max_arity))
        self.halt_w = nn.Parameter(torch.empty(self.n_internal, in_dim))
        self.halt_b = nn.Parameter(torch.full((self.n_internal,), float(split_init)))

        self.expert_w = nn.Parameter(torch.empty(self.n_total, out_dim, in_dim))
        self.expert_b = nn.Parameter(torch.zeros(self.n_total, out_dim))

        self.freeze_structure = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.in_dim)
        nn.init.normal_(self.router_w, std=std)
        nn.init.normal_(self.halt_w, std=std)
        nn.init.uniform_(self.expert_w, -std, std)

    def _level_slice(self, level):
        start = (self.K ** level - 1) // (self.K - 1)
        return start, self.K ** level

    def _lambda(self, x, idx, level):
        if level == self.max_depth:
            return None
        if self.freeze_structure:
            b, width = x.shape[0], self.K ** level
            return x.new_zeros(b, width)
        logits = torch.einsum("ni,bi->bn", self.halt_w[idx], x) + self.halt_b[idx]
        return torch.sigmoid(logits)

    def forward(self, x):
        if not self.training:
            return self.hard_forward(x)[0]
        return self.hard_forward(x)[0]

    def ponder_loss(self, x, y):
        b = x.shape[0]
        reach = x.new_ones(b, 1)
        rec = x.new_zeros(b)
        exp_depth = x.new_zeros(b)
        total_halt = x.new_zeros(b)
        arity_num = x.new_zeros(())
        arity_den = x.new_zeros(()) + 1e-9

        for level in range(self.max_depth + 1):
            start, width = self._level_slice(level)
            idx = slice(start, start + width)

            logits = torch.einsum("noi,bi->bno", self.expert_w[idx], x) + self.expert_b[idx]
            logp = F.log_softmax(logits, dim=-1)
            ce_node = -logp.gather(-1, y[:, None, None].expand(b, width, 1)).squeeze(-1)

            lam = self._lambda(x, idx, level)
            if level == self.max_depth:
                p_halt = reach  
            else:
                p_halt = reach * lam

            rec = rec + (p_halt * ce_node).sum(1)
            exp_depth = exp_depth + p_halt.sum(1) * float(level)
            total_halt = total_halt + p_halt.sum(1)

            if level == self.max_depth:
                break

            cont = reach * (1.0 - lam)              
            rlogits = torch.einsum("nki,bi->bnk", self.router_w[idx], x) + self.router_b[idx]
            r = F.softmax(rlogits, dim=-1)          

            ent = -(r * (r.clamp_min(1e-9)).log()).sum(-1)   
            w_mass = cont                                     
            arity_num = arity_num + (w_mass * ent).sum()
            arity_den = arity_den + w_mass.sum()

            reach = (cont.unsqueeze(-1) * r).reshape(b, width * self.K)

        rec_loss = rec.mean()
        ponder_cost = exp_depth.mean()
        arity_penalty = arity_num / arity_den
        return rec_loss, ponder_cost, arity_penalty

    @torch.no_grad()
    def hard_forward(self, x):
        b = x.shape[0]
        flat = torch.zeros(b, dtype=torch.long, device=x.device)
        alive = torch.ones(b, dtype=torch.bool, device=x.device)
        path_len = torch.zeros(b, device=x.device)

        for _ in range(self.max_depth):
            if self.freeze_structure:
                halt_here = torch.zeros(b, dtype=torch.bool, device=x.device)
            else:
                hl = (self.halt_w[flat] * x).sum(-1) + self.halt_b[flat]
                halt_here = torch.sigmoid(hl) > self.halt_threshold
            do_descend = alive & ~halt_here

            rw = self.router_w[flat]
            logits = torch.einsum("bki,bi->bk", rw, x) + self.router_b[flat]
            k = logits.argmax(-1)
            new_flat = self.K * flat + 1 + k

            flat = torch.where(do_descend, new_flat, flat)
            path_len = path_len + do_descend.float()
            alive = alive & do_descend
            if not alive.any():
                break

        out = torch.bmm(self.expert_w[flat], x.unsqueeze(-1)).squeeze(-1) + self.expert_b[flat]
        return out, path_len

    @torch.no_grad()
    def structure_stats(self, X):
        """Data-dependent structure summary (pass a held-out batch)."""
        self.eval()
        _, path_len = self.hard_forward(X)
        b = X.shape[0]
        flat = torch.zeros(b, dtype=torch.long, device=X.device)
        alive = torch.ones(b, dtype=torch.bool, device=X.device)
        ent_sum, ent_cnt = 0.0, 0
        for _ in range(self.max_depth):
            hl = (self.halt_w[flat] * X).sum(-1) + self.halt_b[flat]
            halt_here = torch.sigmoid(hl) > self.halt_threshold
            do_descend = alive & ~halt_here
            rl = torch.einsum("bki,bi->bk", self.router_w[flat], X) + self.router_b[flat]
            r = F.softmax(rl, dim=-1)
            ent = -(r * r.clamp_min(1e-9).log()).sum(-1)
            ent_sum += float(ent[do_descend].sum()); ent_cnt += int(do_descend.sum())
            k = rl.argmax(-1)
            flat = torch.where(do_descend, self.K * flat + 1 + k, flat)
            alive = alive & do_descend
            if not alive.any():
                break
        eff_arity = math.exp(ent_sum / ent_cnt) if ent_cnt else 0.0
        return {
            "avg_path_len": path_len.mean().item(),
            "path_len_std": path_len.std().item(),
            "frac_max_depth": (path_len >= self.max_depth).float().mean().item(),
            "effective_arity": eff_arity,
            "max_arity": float(self.K),
        }


class MRFFFClassifier(nn.Module):
    def __init__(self, in_dim, n_classes, max_depth, max_arity, **kwargs):
        super().__init__()
        self.mrfff = MultiResolutionFFF(in_dim, n_classes, max_depth, max_arity, **kwargs)

    def forward(self, x):
        return self.mrfff(x)

    def loss(self, x, y, reg_split=0.0, reg_arity=0.0):
        rec, ponder, arity = self.mrfff.ponder_loss(x, y)
        return rec + reg_split * ponder + reg_arity * arity


if __name__ == "__main__":
    torch.manual_seed(0)
    in_dim, n_cls, b = 16, 5, 64
    x = torch.randn(b, in_dim)
    y = torch.randint(0, n_cls, (b,))

    clf = MRFFFClassifier(in_dim, n_cls, max_depth=4, max_arity=3, split_init=-1.0)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
    for step in range(200):
        clf.train()
        opt.zero_grad()
        loss = clf.loss(x, y, reg_split=0.02, reg_arity=0.001)
        loss.backward()
        opt.step()
    clf.eval()
    print("final train loss:", float(loss))
    print("structure:", clf.mrfff.structure_stats(x))
    print("eval logits shape:", clf(x).shape)