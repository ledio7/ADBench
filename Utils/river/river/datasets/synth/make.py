import numpy as np
import itertools
from river.utils.skmultiflow_utils import check_random_state
from sklearn.datasets import make_classification
from .. import base


class Make(base.SyntheticDataset):
    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 20,
        n_informative : int = 2,
        rate:  float = 0.01,
        seed: int or np.random.RandomState = None,
    ):
        super().__init__(
            n_features=n_features, n_classes=2, n_outputs=1, task=base.BINARY_CLF
        )

        self.seed = seed
        self.n_samples = n_samples
        self.n_informative = n_informative
        self.n_features = n_features
        self.rate = rate


    def __iter__(self):
        self.x, self.y = make_classification(
            n_samples=self.n_samples,
            n_features= self.n_features,
            n_informative= self.n_informative,
            n_classes=2,  
            random_state = self.seed, 
            weights= [1 - self.rate],
            n_redundant= 0,
            n_clusters_per_class=1,
            shuffle = False,
        )

        for xi, yi in itertools.zip_longest(
            self.x, self.y if hasattr(self.y, "__iter__") else []
        ):
            yield dict(zip(range(self.n_features), xi)), yi
