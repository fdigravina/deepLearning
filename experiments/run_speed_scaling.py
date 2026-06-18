import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import config as C
from fff import FFFClassifier, MLPClassifier, fff_flops, mlp_flops, set_seed, time_inference
from fff.utils import Logger

IN_DIM, OUT_DIM, BATCH, SEED = 64, 10, 1024, 0
DEPTHS = list(range(4, 13))

def run():
    log = Logger(os.path.join(C.RESULTS_DIR, "speed_scaling.txt"))
    log(f"speed scaling | in={IN_DIM} out={OUT_DIM} batch={BATCH}")
    log(f"{'width':<8} {'depth':<10} {'FFF_us':<12} {'MLP_us':<12} {'speedup':<10} {'flop_x':<10}")
    set_seed(SEED)
    x = torch.randn(BATCH, IN_DIM)
    widths, t_fff, t_mlp, fl_fff, fl_mlp = [], [], [], [], []
    for d in DEPTHS:
        w = 2 ** d
        fff = FFFClassifier(IN_DIM, OUT_DIM, depth=d)
        mlp = MLPClassifier(IN_DIM, OUT_DIM, hidden=w)
        tf = time_inference(fff, x)
        tm = time_inference(mlp, x)
        ff, fm = fff_flops(IN_DIM, OUT_DIM, d), mlp_flops(IN_DIM, w, OUT_DIM)
        widths.append(w); t_fff.append(tf*1e6); t_mlp.append(tm*1e6); fl_fff.append(ff); fl_mlp.append(fm)
        log(f"{w:<8} {d:<10} {tf*1e6:<12.3f} {tm*1e6:<12.3f} {tm/tf:<10.1f} {fm/ff:<10.1f}")
    _plot(widths, t_fff, t_mlp, fl_fff, fl_mlp)
    log.close()

def _plot(widths, t_fff, t_mlp, fl_fff, fl_mlp):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(widths, t_mlp, "o-", label="MLP")
    ax.plot(widths, t_fff, "s-", label="FFF")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_ylabel("time/us")
    ax.set_title("Inference time scaling"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "speed_scaling_time.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.plot(widths, fl_mlp, "o-", label="MLP")
    ax.plot(widths, fl_fff, "s-", label="FFF")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_ylabel("FLOPs")
    ax.set_title("Analytical FLOPs"); ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "speed_scaling_flops.png"), dpi=130); plt.close(fig)

if __name__ == "__main__":
    run()