import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
from experiments.common import evaluate, get_dataset, train_model
from fff import FFFClassifier, MLPClassifier, MRFFFClassifier, count_parameters, fff_flops, mlp_flops, set_seed, time_inference
from fff.utils import Logger

def run():
    log = Logger(os.path.join(C.RESULTS_DIR, "benchmark.txt"))
    log(f"FFF vs MLP vs MR-FFF | seed={C.SEED} | depth={C.BENCH_DEPTH} ({2**C.BENCH_DEPTH} leaves)")
    results = {}
    for name in C.DATASETS:
        set_seed(C.SEED)
        Xtr, ytr, Xte, yte, in_dim, n_cls = get_dataset(name, seed=C.SEED)
        width = 2 ** C.BENCH_DEPTH
        bs = C.BATCH_SIZE_LARGE if name in ["adult", "mnist_784", "fashion_mnist"] else C.BATCH_SIZE

        fff = FFFClassifier(in_dim, n_cls, depth=C.BENCH_DEPTH)
        train_model(fff, Xtr, ytr, epochs=C.EPOCHS, batch_size=bs, seed=C.SEED)
        acc_fff, t_fff = evaluate(fff, Xte, yte), time_inference(fff, Xte)

        set_seed(C.SEED)
        mlp1 = MLPClassifier(in_dim, n_cls, hidden=width, n_layers=1)
        train_model(mlp1, Xtr, ytr, epochs=C.EPOCHS, batch_size=bs, seed=C.SEED)
        acc_mlp1, t_mlp1 = evaluate(mlp1, Xte, yte), time_inference(mlp1, Xte)

        set_seed(C.SEED)
        mlp2 = MLPClassifier(in_dim, n_cls, hidden=width, n_layers=2)
        train_model(mlp2, Xtr, ytr, epochs=C.EPOCHS, batch_size=bs, seed=C.SEED)
        acc_mlp2, t_mlp2 = evaluate(mlp2, Xte, yte), time_inference(mlp2, Xte)

        set_seed(C.SEED)
        mrfff = MRFFFClassifier(in_dim, n_cls, max_depth=5, max_arity=4, split_init=-1.0)
        train_model(mrfff, Xtr, ytr, epochs=C.EPOCHS, batch_size=bs,
                    reg_arity=0.001, reg_split=0.02, lr=1e-3, seed=C.SEED)
        acc_mrfff, t_mrfff = evaluate(mrfff, Xte, yte), time_inference(mrfff, Xte)
        path_mrfff = mrfff.mrfff.structure_stats(Xte)["avg_path_len"]

        fl_fff = fff_flops(in_dim, n_cls, C.BENCH_DEPTH)
        fl_mlp1 = mlp_flops(in_dim, width, n_cls, n_layers=1)
        fl_mlp2 = mlp_flops(in_dim, width, n_cls, n_layers=2)

        results[name] = {
            "acc_fff": acc_fff,
            "acc_mlp1": acc_mlp1,
            "acc_mlp2": acc_mlp2,
            "acc_mrfff": acc_mrfff,
            "t_fff": t_fff,
            "t_mlp1": t_mlp1,
            "t_mlp2": t_mlp2,
            "t_mrfff": t_mrfff,
            "speedup_1": t_mlp1 / t_fff,
            "speedup_2": t_mlp2 / t_fff,
            "time_ratio_mr": t_mrfff / t_fff,
            "flop_ratio_1": fl_mlp1 / fl_fff,
            "flop_ratio_2": fl_mlp2 / fl_fff,
            "path_mrfff": path_mrfff,
            "p_fff": count_parameters(fff),
            "p_mlp1": count_parameters(mlp1),
            "p_mlp2": count_parameters(mlp2),
            "p_mrfff": count_parameters(mrfff),
        }
        r = results[name]
        log(f"\n[{name}] in={in_dim} cls={n_cls} test={len(yte)}")
        log(f"  acc  FFF={acc_fff:.4f}  MLP1={acc_mlp1:.4f}  MLP2={acc_mlp2:.4f}  MR-FFF={acc_mrfff:.4f}")
        log(f"  infer  FFF={t_fff*1e6:.2f}us  MLP1={t_mlp1*1e6:.2f}us  MLP2={t_mlp2*1e6:.2f}us  MR-FFF={t_mrfff*1e6:.2f}us")
        log(f"  speedup vs MLP1={r['speedup_1']:.1f}x  vs MLP2={r['speedup_2']:.1f}x")
        log(f"  MR-FFF time vs FFF={r['time_ratio_mr']:.2f}x (>1 = slower)  avg_path={path_mrfff:.2f}/5")
        log(f"  FLOP ratio vs MLP1={r['flop_ratio_1']:.1f}x  vs MLP2={r['flop_ratio_2']:.1f}x")
        log(f"  params  FFF={r['p_fff']}  MLP1={r['p_mlp1']}  MLP2={r['p_mlp2']}  MR-FFF={r['p_mrfff']}")
    _plot(results)
    log.close()
    return results

def _plot(results):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    names = list(results)
    x = range(len(names))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.2
    for i, (k, lbl) in enumerate([("acc_fff", "FFF"), ("acc_mlp1", "MLP1"), ("acc_mlp2", "MLP2"), ("acc_mrfff", "MR-FFF")]):
        ax.bar([xi + (i - 1.5) * w for xi in x], [results[n][k] for n in names], w, label=lbl)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("test accuracy")
    ax.set_title("Predictive performance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "benchmark_accuracy.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([xi - 0.15 for xi in x], [results[n]["speedup_1"] for n in names], 0.3, label="MLP1 / FFF")
    ax.bar([xi + 0.15 for xi in x], [results[n]["speedup_2"] for n in names], 0.3, label="MLP2 / FFF")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("speedup over FFF (x, higher = better)")
    ax.set_title("FFF inference speedup vs dense MLPs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "benchmark_efficiency.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(x), [results[n]["time_ratio_mr"] for n in names], 0.5, color="tab:purple")
    ax.set_xticks(list(x)); ax.set_xticklabels(names)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_ylabel("MR-FFF time / FFF time (x, lower = better)")
    ax.set_title("MR-FFF inference cost relative to FFF")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "benchmark_mrfff_cost.png"), dpi=130)
    plt.close(fig)

if __name__ == "__main__":
    run()