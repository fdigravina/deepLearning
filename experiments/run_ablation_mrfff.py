import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import config as C
from experiments.common import evaluate, get_dataset, train_model
from fff import FFFClassifier, MRFFFClassifier, count_parameters, set_seed, time_inference
from fff.utils import Logger

SEED = 0
FIXED = [(2, 6), (4, 3), (8, 2)]
LEARNED_ARITY, LEARNED_DEPTH = 4, 3
REG_SPLIT = [0.0, 0.02, 0.1]
REG_ARITY = 0.001

@torch.no_grad()
def avg_path_length(model, X):
    model.eval()
    _, path_len = model.mrfff.hard_forward(X)
    return path_len.mean().item()

def run():
    log = Logger(os.path.join(C.RESULTS_DIR, "ablation.txt"))
    log(f"MR-FFF ablation | seed={SEED}")
    all_res = {}
    for name in C.DATASETS:
        set_seed(SEED)
        Xtr, ytr, Xte, yte, in_dim, n_cls = get_dataset(name, seed=SEED)
        log(f"\n=== {name} (in={in_dim}, cls={n_cls}) ===")

        set_seed(SEED)
        ref = FFFClassifier(in_dim, n_cls, depth=6)
        train_model(ref, Xtr, ytr, epochs=C.EPOCHS, seed=SEED)
        log(f"ref FFF depth=6  acc={evaluate(ref,Xte,yte):.4f}  time={time_inference(ref,Xte)*1e6:.2f}us  params={count_parameters(ref)}")

        log("\n-- fixed arity (frozen full-depth K-ary tree) --")
        fixed = []
        for k, d in FIXED:
            set_seed(SEED)
            m = MRFFFClassifier(in_dim, n_cls, max_depth=d, max_arity=k)
            m.mrfff.freeze_structure = True
            train_model(m, Xtr, ytr, epochs=C.EPOCHS, seed=SEED)
            acc = evaluate(m, Xte, yte)
            t = time_inference(m, Xte) * 1e6
            pl = avg_path_length(m, Xte)
            fixed.append({"arity": k, "depth": d, "acc": acc, "time": t, "path": pl,
                          "params": count_parameters(m)})
            log(f"arity={k} depth={d}  acc={acc:.4f}  time={t:.2f}us  path={pl:.2f}  params={count_parameters(m)}")

        log("\n-- learned arity/depth (input-dependent halting) --")
        learned = []
        for rs in REG_SPLIT:
            set_seed(SEED)
            m = MRFFFClassifier(in_dim, n_cls, max_depth=LEARNED_DEPTH, max_arity=LEARNED_ARITY,
                                split_init=-1.0)
            train_model(m, Xtr, ytr, reg_arity=REG_ARITY, reg_split=rs, epochs=C.EPOCHS, seed=SEED)
            acc = evaluate(m, Xte, yte)
            t = time_inference(m, Xte) * 1e6
            st = m.mrfff.structure_stats(Xte)
            learned.append({"reg": rs, "acc": acc, "time": t,
                            "path": st["avg_path_len"], "path_std": st["path_len_std"],
                            "frac_max": st["frac_max_depth"],
                            "effective_arity": st["effective_arity"]})
            log(f"reg_split={rs:.3f}  acc={acc:.4f}  time={t:.2f}us  path={st['avg_path_len']:.2f}"
                f"(+/-{st['path_len_std']:.2f})  frac_max_depth={st['frac_max_depth']:.2f}"
                f"  eff_arity={st['effective_arity']:.2f}")
        all_res[name] = {"fixed": fixed, "learned": learned}
    _plot(all_res)
    log.close()

def _plot(all_res):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    for name, res in all_res.items():
        f = res["fixed"]
        ar = [r["arity"] for r in f]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(ar, [r["acc"] for r in f], "o-", color="tab:blue", label="accuracy")
        ax2.plot(ar, [r["time"] for r in f], "s--", color="tab:red", label="time")
        ax1.set_xlabel("fixed arity"); ax1.set_ylabel("accuracy", color="tab:blue")
        ax2.set_ylabel("time/us", color="tab:red")
        ax1.set_xticks(ar); ax1.set_title(f"MR-FFF fixed arity ({name})")
        fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, f"ablation_fixed_arity_{name}.png"), dpi=130); plt.close(fig)

        l = res["learned"]
        reg = [r["reg"] for r in l]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax2 = ax1.twinx()
        ax1.plot(reg, [r["path"] for r in l], "^-", color="tab:olive", label="avg path len")
        ax1.plot(reg, [r["effective_arity"] for r in l], "o-", color="tab:green", label="eff arity")
        ax2.plot(reg, [r["acc"] for r in l], "s--", color="tab:blue", label="accuracy")
        ax1.set_xlabel("ponder penalty (reg_split)")
        ax1.set_ylabel("learned structure", color="tab:green")
        ax2.set_ylabel("accuracy", color="tab:blue")
        ax1.set_title(f"MR-FFF learned ({name})")
        h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, f"ablation_learned_{name}.png"), dpi=130); plt.close(fig)

if __name__ == "__main__":
    run()