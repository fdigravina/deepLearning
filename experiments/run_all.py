import os
import time
import config as C
from experiments import run_benchmark, run_speed_scaling, run_ablation_mrfff
from fff.utils import Logger

def run():
    log = Logger(os.path.join(C.RESULTS_DIR, "summary.txt"))
    log("FFF experiment suite")
    log("=" * 40)
    for name, fn in [("benchmark", run_benchmark.run),
                     ("speed_scaling", run_speed_scaling.run),
                     ("ablation", run_ablation_mrfff.run)]:
        t0 = time.time()
        log(f"\n>>> {name}")
        fn()
        log(f">>> {name} done in {time.time()-t0:.1f}s")
    log("\nAll results saved to results/ (txt) and results/figures/ (png)")
    log.close()

if __name__ == "__main__":
    run()