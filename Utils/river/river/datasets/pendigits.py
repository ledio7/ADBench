from river import stream

from . import base


class Pendigits(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=6_870,
            n_features=16,
            filename="pendigits.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 16)}
        self.converters["16"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="16", converters=self.converters)