from river import stream

from . import base


class Wbc(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=278,
            n_features=30,
            filename="wbc.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 30)}
        self.converters["30"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="30", converters=self.converters)