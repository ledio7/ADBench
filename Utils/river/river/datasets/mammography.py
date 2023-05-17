from river import stream

from . import base


class Mammography(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=11_183,
            n_features=6,
            filename="mammography.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 6)}
        self.converters["6"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="6", converters=self.converters)