from river import stream

from . import base


class Vowels(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=1_456,
            n_features=12,
            filename="vowels.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 12)}
        self.converters["12"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="12", converters=self.converters)