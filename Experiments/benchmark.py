import pandas as pd
import multiprocessing as mp
import pathlib
from evaluate import aggregate_dataframe, test_then_train
from main_plots import create_main_plots, create_curves
from tqdm import tqdm
import os
import sys
# sys.setrecursionlimit(100_000)

N_PROCESSES = 6
DATASETS = ["arrhythmia", "satimage-2", "musk" ]
# DATASETS = ["annthyroid", "arrhythmia", "breastw", "cardio", "letter", "mammography", "mnist", "musk", "optdigits", "pendigits", "speech", "thyroid", "vowels", "wbc"]
# DATASETS = ["satimage-2", "annthyroid", "cardio", "shuttle"]

# MODELS = ["AE", "DAE", "PW-AE", "RRCF", "HST", "xStream", "ILOF", "LODA"]
# MODELS = ["AE", "DAE", "PW-AE", "HST", "ILOF", "LODA"] #new one
MODELS = ["AE", "DAE", "HST"]
SEEDS = range(42, 47)

if len(sys.argv) < 2:
        SUBSAMPLE=600_000
else: 
        SUBSAMPLE = sys.argv[1]

# SUBSAMPLE = 600_000


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
        rates = []
        
        for run in runs:
            metrics.append(run.get()[0])
            rates.append(run.get()[1])
            pbar.update(1)

    metrics_raw = pd.DataFrame(metrics)
    metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])

    path = pathlib.Path(__file__).parent.parent.resolve()

    # to_raw = os.path.join('Experiments', 'Results', 'Benchmark_raw.csv')
    path_raw = path.joinpath(os.path.join('Experiments', 'Results', 'Benchmark_raw.csv'))

    # to_aggregate = os.path.join('Experiments', 'Results', 'Benchmark_agg.csv')
    path_agg = path.joinpath(os.path.join('Experiments', 'Results', 'Benchmark_agg.csv'))

    metrics_raw.to_csv(path_raw)
    metrics_agg.to_csv(path_agg)
    
    #Create roc e pr curve
    column_names = ['model', 'dataset', 'seed', 'fpr', 'tpr' ]
    df_rates= pd.DataFrame(rates, columns=column_names)
    # print(df_rates['tpr'][1])
    create_curves(df_rates)

    #Create main heatmaps
    create_main_plots(metrics_agg)

    pool.close()
    pool.join()
