import pandas as pd
import multiprocessing as mp
import pathlib
from evaluation import aggregate_dataframe, test_then_train
from main_plots import create_main_plots, create_curves 
from tqdm import tqdm
import os
import sys
import ast
import numpy as np

N_PROCESSES = 7
# DATASETS = ["covertype","satimage-2", "shuttle","mammography", "thyroid", "creditcard"]
DATASETS = [
            "covertype","satimage-2", "shuttle", "http", "smtp", "creditcard",
            "annthyroid", "arrhythmia", "breastw", "cardio", "letter", 
            "mammography", "mnist", "musk", "optdigits", "pendigits", 
            "speech", "thyroid", "vowels", "wbc"
            ]
# DATASETS = ["satimage-2", "mammography", "vowels", "optdigits", "pendigits", "wbc"]


MODELS = ["AE", "DAE", "PW-AE", "HST", "xStream", "ILOF", "LODA", "RRCF"]
# MODELS = ["AE", "DAE", "PW-AE", "HST", "ILOF", "LODA"] #new one
# MODELS = ["AE"]
SEEDS = range(42, 47)

if len(sys.argv) < 2:
        SUBSAMPLE=600_000
else: 
        SUBSAMPLE = int(sys.argv[1])

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

    with tqdm(total=len(runs), desc="Processing") as pbar:
        metrics = []
        rates = []
        
        for run in runs:
            metrics.append(run.get()[0])
            rates.append(run.get()[1])
            pbar.update(1)

    metrics_raw = pd.DataFrame(metrics)
    metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])
    column_names = ['model', 'dataset', 'seed', 'fpr', 'tpr', 'recall', 'precision']
    df_rates= pd.DataFrame(rates, columns=column_names)

    path_raw = os.path.join('Results', 'Benchmark_raw.csv')
    path_agg = os.path.join('Results', 'Benchmark_agg.csv')
    path_rate = os.path.join('Results', 'Rates.csv')

    metrics_raw.to_csv(path_raw)
    metrics_agg.to_csv(path_agg)
    np.set_printoptions(threshold=np.inf)
    df_rates.to_csv(path_rate, index=False)

    df_rates = pd.read_csv(path_rate)
    for metric in column_names[3:]:
        df_rates[metric] = df_rates[metric].str.strip('[]').str.split()
        df_rates[metric] = df_rates[metric].apply(lambda x: np.array(x, dtype=float))

    # Create pr and roc curves
    create_curves(df_rates)
    
    # Create main heatmaps
    create_main_plots(metrics_agg)

    pool.close()
    pool.join()