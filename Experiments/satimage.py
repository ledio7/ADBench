from river import stream

from IncrementalTorch.datasets import base


class Satimage(base.FileDataset):
    """Forest cover type information.

    Subsample of the original covertype dataset modified for anomaly detection tasks. Class 2 of the original dataset is labeled as normal, class 4 as anomalous. Contains 2747 

    """

    def __init__(self):
        super().__init__(
            n_samples=5_803,
            n_features=36,
            filename="satimage-2.csv.",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 36)}
        self.converters["y"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="y", converters=self.converters)