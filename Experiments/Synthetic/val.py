import os
import random
import sys
import warnings
import numpy as np
from IncrementalTorch.anomaly import *
from IncrementalTorch.base import AutoencoderBase
from river.anomaly import *
from river.datasets import *
from river.preprocessing import AdaptiveStandardScaler, MinMaxScaler, Normalizer
from river.feature_extraction import RBFSampler
import torch
from streamad.model import LodaDetector
from time import time
from river.datasets import synth
from metrics import compute_metrics
import pandas as pd
from river import stream
from data_adapter import handle_dataset, get_converters
import pandas as pd
import multiprocessing as mp
import pathlib
from main_plots import create_main_plots
from tqdm import tqdm
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

N_PROCESSES = 5
SEEDS = range(42, 44)


DATASETS = {
    # "agra1" : synth.Agrawal(classification_function=5, seed=42, balance_classes=True, perturbation = 0.1 ).take(20000),
    # "agra2" : synth.Agrawal(classification_function=5, seed=43, balance_classes=True, perturbation = 0.1 ).take(100000),
    # "agra3" : synth.Agrawal(classification_function=5, seed=44, balance_classes=True, perturbation = 0.1 ).take(200000),
    # "agra4" : synth.Agrawal(classification_function=5, seed=45, balance_classes=True, perturbation = 0.1 ).take(200000), 
    # "agra5" : synth.Agrawal(classification_function=5, seed=46, balance_classes=True, perturbation = 0.2 ).take(200000),

    # "agra6" : synth.Agrawal(classification_function=9, seed=47, balance_classes=True, perturbation = 0.1 ).take(20000),   
    # "agra7" : synth.Agrawal(classification_function=9, seed=48, balance_classes=True, perturbation = 0.1 ).take(100000),   
    # "agra8" : synth.Agrawal(classification_function=9, seed=49, balance_classes=True, perturbation = 0.1 ).take(200000),   
    # "agra9" : synth.Agrawal(classification_function=9, seed=50, balance_classes=True, perturbation = 0.1 ).take(200000),   
    # "agra10" : synth.Agrawal(classification_function=9, seed=51, balance_classes=True, perturbation = 0.2 ).take(200000),    

    # "sine1": synth.AnomalySine(n_samples=10000, n_anomalies=100, replace= False, noise = 0.1, seed = 43),
    # "sine2": synth.AnomalySine(n_samples=50000, n_anomalies=200, replace= False, noise = 0.1, seed = 43),
    # "sine3": synth.AnomalySine(n_samples=100000, n_anomalies=800, replace= False, noise = 0.1, seed = 43),
    # "sine4": synth.AnomalySine(n_samples=100000, n_anomalies=1200, replace= False, noise = 0.1, seed = 43),
    "sine5": synth.AnomalySine(n_samples=100000, n_anomalies=1800, replace= False, noise = 0.1, seed = 43),

    # "hyper1" : synth.Hyperplane(seed=46, n_features=50, noise_percentage = 0.1).take(10000), 
    # "hyper2" : synth.Hyperplane(seed=46, n_features=100, noise_percentage = 0.1).take(50000), 
    # "hyper3" : synth.Hyperplane(seed=46, n_features=200, noise_percentage = 0.1).take(200000), 
    # "hyper4" : synth.Hyperplane(seed=46, n_features=300, noise_percentage = 0.1).take(200000), 
    # "hyper5" : synth.Hyperplane(seed=46, n_features=400, noise_percentage = 0.1).take(200000), 

    # "tree1" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=50, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(20000),
    # "tree2" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=100, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(100000),
    # "tree3" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=150, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    # "tree4" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=200, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    # "tree5" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=250, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    
    # "make1" : synth.Make(n_samples=10000, n_features=200, n_informative=100, rate = 0.002, seed=45),
    # "make2" : synth.Make(n_samples=50000, n_features=200, n_informative=190, rate = 0.004, seed=45),
    # "make3" : synth.Make(n_samples=100000, n_features=300, n_informative=300, rate = 0.008, seed=45),
    # "make4" : synth.Make(n_samples=100000, n_features=400, n_informative=250, rate = 0.01, seed=45),
    # "make5" : synth.Make(n_samples=100000, n_features=400, n_informative=390, rate = 0.016, seed=45),
    # "sine1": synth.AnomalySine(n_samples=10000, n_anomalies=200, replace= False, noise = 0.3, seed = 42),
    # "sine2": synth.AnomalySine(n_samples=10000, n_anomalies=400, replace= False, noise = 0.3, seed = 43),
    # "sine3": synth.AnomalySine(n_samples=10000, n_anomalies=800, replace= False, noise = 0.3, seed = 44),
    # "sine4": synth.AnomalySine(n_samples=10000, n_anomalies=1200, replace= False, noise = 0.3, seed = 45),
    # "sine5": synth.AnomalySine(n_samples=10000, n_anomalies=1800, replace= False, noise = 0.3, seed = 46),

    # "agra1" : synth.Agrawal(classification_function=5, seed=42, balance_classes=True, perturbation = 0.1 ).take(20000),
    # "agra2" : synth.Agrawal(classification_function=5, seed=43, balance_classes=True, perturbation = 0.1 ).take(20000),
    # "agra3" : synth.Agrawal(classification_function=5, seed=44, balance_classes=True, perturbation = 0.1 ).take(20000),
    # "agra4" : synth.Agrawal(classification_function=5, seed=45, balance_classes=True, perturbation = 0.1 ).take(20000), 
    # "agra5" : synth.Agrawal(classification_function=5, seed=46, balance_classes=True, perturbation = 0.2 ).take(20000),

    # "agra6" : synth.Agrawal(classification_function=9, seed=47, balance_classes=True, perturbation = 0.1 ).take(2000),   
    # "agra7" : synth.Agrawal(classification_function=9, seed=48, balance_classes=True, perturbation = 0.1 ).take(2000),   
    # "agra8" : synth.Agrawal(classification_function=9, seed=49, balance_classes=True, perturbation = 0.1 ).take(2000),   
    # "agra9" : synth.Agrawal(classification_function=9, seed=50, balance_classes=True, perturbation = 0.1 ).take(2000),   
    # "agra10" : synth.Agrawal(classification_function=9, seed=51, balance_classes=True, perturbation = 0.2 ).take(2000),   

    # "hyper1" : synth.Hyperplane(seed=46, n_features=20, noise_percentage = 0.1).take(20000), 
    # "hyper2" : synth.Hyperplane(seed=46, n_features=20, noise_percentage = 0.1).take(20000), 
    # "hyper3" : synth.Hyperplane(seed=46, n_features=20, noise_percentage = 0.1).take(20000), 
    # "hyper4" : synth.Hyperplane(seed=46, n_features=20, noise_percentage = 0.1).take(20000), 
    # "hyper5" : synth.Hyperplane(seed=46, n_features=20, noise_percentage = 0.1).take(20000), 

    # "tree1" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=20, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(20000),
    # "tree2" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=20, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    # "tree3" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=20, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    # "tree4" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=20, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    # "tree5" : synth.RandomTree(seed_tree=45, seed_sample=42, n_num_features=20, n_cat_features= 0, max_tree_depth=18, first_leaf_level=15).take(200000),
    
    # "make1" : synth.Make(n_samples=10000, n_features=20, n_informative=20, rate = 0.001, seed=42),
    # "make2" : synth.Make(n_samples=10000, n_features=20, n_informative=20, rate = 0.0025, seed=43),
    # "make3" : synth.Make(n_samples=10000, n_features=20, n_informative=20, rate = 0.007, seed=44),
    # "make4" : synth.Make(n_samples=10000, n_features=20, n_informative=20, rate = 0.009, seed=45),
    # "make5" : synth.Make(n_samples=10000, n_features=20, n_informative=20, rate = 0.013, seed=46),
}

PREPROCESSORS = {
    "minmax": MinMaxScaler,
    "standard": AdaptiveStandardScaler,
    "norm": Normalizer,
    "rbf": RBFSampler,
    "none": None,
}


MODELS = {
    # "DAE": AutoencoderBase,
    "AE": NoDropoutAE,
    # "PW-AE": ProbabilityWeightedAutoencoder,
    # "xStream": xStream,
    # "RRCF": RobustRandomCutForest,
    # "ILOF": ILOF,
    # "HST": HalfSpaceTrees,
    # "LODA": LodaDetector
}

CONFIGS = {
    "AE": {"lr": 0.02, "latent_dim": 0.1},
    "DAE": {"lr": 0.02},
    "PW-AE": {"lr": 0.1},
    "HST": {"n_trees": 25, "height": 15},
    "LODA": {"window_len" : 256}
}

def test_then_train(
    # data,
    dataset_name,
    # model,
    model_name,
    update_interv=1000,
    log_memory=False,
    preprocessor="minmax",
    seed=None,
    **model_kwargs,
):
    func_kwargs = dict( #names of the columns
        model=model_name,
        dataset = dataset_name,
        preprocessor=preprocessor,
        seed=seed,
        **model_kwargs,
    )
    
    if seed:
        seed_everything(seed)
    
    
    total_time = 0
    
    # Initialize preprocessor
    try:
        _preprocessor = PREPROCESSORS[preprocessor]
        if _preprocessor:
            _preprocessor = _preprocessor()
    except KeyError:
        _preprocessor = None
        warnings.warn(f"Preprocessor '{preprocessor}' could not be found.")


    # Initialize model
    if isinstance(model_name, str):
        model = MODELS[model_name]
        model = model(**model_kwargs)
    

    scores, labels = [], []
    converters = get_converters(dataset_name)
    anomalies = 0
    samples = 0
    start = time()
    for x, y in stream.iter_csv(os.path.join('Data_gen', f'{dataset_name}.csv'), target= 'y', converters= converters):
        samples +=1
        if y ==1:
            anomalies +=1

        # Preprocess input
        if _preprocessor:
            _preprocessor.learn_one(x)
            x = _preprocessor.transform_one(x)

        #LODA version
        if model_name == "LODA":
            x = np.array(list(x.values()))
            score = model.fit_score(x)
            if score is not None:   
                scores.append(score)
                labels.append(y)
            
        else: 
            score = model.score_learn_one(x)

            scores.append(score)
            labels.append(y)

    n_features = len(x) 

    # Compute final metric scores
    total_time += time() - start

    metrics = compute_metrics(labels, scores)
    
    metrics["runtime"] = total_time
    metrics["samples"] = samples
    metrics["anomalies"]= anomalies
    metrics["n_features"] = n_features

    metrics.update(func_kwargs)

    return metrics


def aggregate_dataframe(df, variables):
    grouped = df.groupby(variables)
    means = grouped.mean()
    stds = grouped.std()
    stds.columns = [f"{column}_std" for column in stds.columns]
    df_summary = means.join(stds, on=variables)
    return df_summary


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)



if __name__ == '__main__':
    mp.freeze_support()

    # Initialize datasets
    for dataset_name, dataset in DATASETS.items():
        if not os.path.isfile(os.path.join('Data_gen', f'{dataset_name}.csv')):
            print("Creating dataset")
            data = list(DATASETS[dataset_name])
            handle_dataset(dataset_name, data)
        else:
            print("Using existing dataset")
           
    # Execution
    pool = mp.Pool(processes=N_PROCESSES)
    runs = [
        pool.apply_async(
            test_then_train,
            kwds=dict(
                # data=dataset,
                dataset_name=dataset_name,
                # model=model,
                model_name=model_name,
                seed=seed,
                **CONFIGS.get(model_name, {}),
            ),
        )
        for dataset_name, dataset in DATASETS.items()
        for model_name, model in MODELS.items()
        for seed in SEEDS
    ]

    # Storing Results
    with tqdm(total=len(runs), desc="Processing") as pbar:
        metrics = []

        for run in runs:
            metrics.append(run.get())
            print(f"{run.get()['dataset']}, {run.get()['model']} -> Done")
            pbar.update(1)

    metrics_raw = pd.DataFrame(metrics)
    metrics_agg = aggregate_dataframe(metrics_raw, ["dataset", "model"])
    
    path_raw = os.path.join('Benchmark_raw.csv')
    
    path_agg = os.path.join('Benchmark_agg.csv')

    metrics_raw.to_csv(path_raw)
    metrics_agg.to_csv(path_agg)

    create_main_plots(metrics_agg)

    pool.close()
    pool.join()
