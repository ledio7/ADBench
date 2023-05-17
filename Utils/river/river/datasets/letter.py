from river import stream

from . import base


class Letter(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=1_600,
            n_features=32,
            filename="letter.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 32)}
        self.converters["32"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="32", converters=self.converters)