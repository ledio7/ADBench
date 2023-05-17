from river import stream

from . import base


class Optdigits(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=5_216,
            n_features=64,
            filename="optdigits.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 64)}
        self.converters["64"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="64", converters=self.converters)