import pandas as pd
import multiprocessing as mp
import pathlib
from evaluate import aggregate_dataframe, test_then_train
from tqdm import tqdm

N_PROCESSES = 6
DATASETS = ["covertype", "creditcard", "shuttle", "satimage-2", "smtp", "http"]
# DATASETS = ["creditcard", "satimage-2", "shuttle"]
# MODELS = ["AE", "HST", "LODA"]
MODELS = ["AE", "DAE", "PW-AE", "RRCF", "HST", "xStream", "ILOF", "LODA"]
SEEDS = range(42, 44)

# SUBSAMPLE = 500_000
SUBSAMPLE = 3001

SAVE_STR = "Benchmark"

CONFIGS = {
    "AE": {"lr": 0.02, "latent_dim": 0.1},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "OC-SVM": {},
    "HST": {"n_trees": 25, "height": 15},
    "LODA": {"window_len" : 256}
}

if __name__ == '__main__':
    mp.freeze_support()

    pool = mp.Pool(processes=N_PROCESSES)
    runs = [
        pool.apply_async(
            test_then_train,
            kwds=dict(
                dataset=dataset,
                model=model,
                seed=seed,
                subsample=SUBSAMPLE,
                **CONFIGS.get(model, {}),
            ),
        )
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
    ]

    # metrics = [run.get()[0] for run in runs]
    with tqdm(total=len(runs), desc="Processing") as pbar:
        metrics = []
        for run in runs:
            metrics.append(run.get()[0])
            pbar.update(1)
    
    metrics_raw = pd.DataFrame(metrics)
    metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

    path = pathlib.Path(__file__).parent.parent.resolve()
    metrics_raw.to_csv(f"{path}\\Experiments\\Results\\{SAVE_STR}_raw.csv")
    metrics_agg.to_csv(f"{path}\\Experiments\\Results\\{SAVE_STR}.csv")


