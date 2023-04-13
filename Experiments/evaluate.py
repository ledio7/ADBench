import os
import random
import sys
import warnings
from river.anomaly.rs_hash import RSHash
import numpy as np
from IncrementalTorch.anomaly import *
from IncrementalTorch.base import AutoencoderBase
from IncrementalTorch.datasets import MNIST, Covertype, Shuttle
from river.anomaly import *
from river.datasets import HTTP, SMTP, CreditCard
from river.preprocessing import AdaptiveStandardScaler, MinMaxScaler, Normalizer
from river.feature_extraction import RBFSampler
import torch
from satimage import Satimage
from streamad.model import LodaDetector
from time import time
# from tqdm import tqdm
from metrics import compute_metrics

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))



DATASETS = {
    "covertype": Covertype,
    "creditcard": CreditCard,
    "shuttle": Shuttle,
    "mnist": MNIST,
    "smtp": SMTP,
    "http": HTTP,
    "satimage-2": Satimage,
}

PREPROCESSORS = {
    "minmax": MinMaxScaler,
    "standard": AdaptiveStandardScaler,
    "norm": Normalizer,
    "rbf": RBFSampler,
    "none": None,
}


MODELS = {
    "DAE": AutoencoderBase,
    "AE": NoDropoutAE,
    "RW-AE": RollingWindowAutoencoder,
    "PW-AE": ProbabilityWeightedAutoencoder,
    "Kit-Net": KitNet,
    "xStream": xStream,
    "RRCF": RobustRandomCutForest,
    "ILOF": ILOF,
    "OC-SVM": OneClassSVM,
    "HST": HalfSpaceTrees,
    "VAE": VariationalAutoencoder,
    "rshash": RSHash,
    "LODA": LodaDetector
}


def test_then_train(
    dataset,
    model,
    subsample=50000,
    update_interv=1000,
    log_memory=False,
    preprocessor="minmax",
    postprocessor="none",
    seed=None,
    **model_kwargs,
):
    func_kwargs = dict(
        model=model,
        subsample=subsample,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        seed=seed,
        **model_kwargs,
    )
    
    if seed:
        seed_everything(seed)
    
    # Get data
    if isinstance(dataset, str):
        if dataset not in DATASETS:
            assert f"Dataset '{dataset}' could not be found."
        else:
            data = list(DATASETS[dataset]().take(subsample))
            func_kwargs["dataset"] = dataset
    else:
        data = dataset
    
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
    if isinstance(model, str):
        try:
            model = MODELS[model]
            model = model(**model_kwargs)
        except KeyError:
            warnings.warn(f"Model '{model}' could not be found.")

    scores, labels = [], []
    start = time()
    # for idx, (x, y) in enumerate(tqdm(data)):
    for idx, (x, y) in enumerate(data):
        # Preprocess input
        if _preprocessor:
            _preprocessor.learn_one(x)
            x = _preprocessor.transform_one(x)

        #LODA version
        if isinstance(model, MODELS["LODA"]):
            x = np.array(list(x.values()))
            score = model.fit_score(x)
            if score is not None:   
                scores.append(score)
                labels.append(y)
            
        else: 
            score = model.score_learn_one(x)
            # Add results
            scores.append(score)
            labels.append(y)
        
        

    # Compute final metric scores
    total_time += time() - start

    metrics = compute_metrics(labels, scores)
    metrics["runtime"] = total_time

    metrics["status"] = "Completed"
    metrics.update(func_kwargs)
    # if isinstance(model, MODELS["LODA"]):
    #     print(f"LODA, {dataset}, {seed}, Done")
    # else:
    #     print(f"{model}, {dataset}, {seed}, Done")
    return metrics, scores


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


