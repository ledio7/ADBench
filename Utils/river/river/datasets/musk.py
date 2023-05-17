from river import stream

from . import base


class Musk(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=3_062,
            n_features=166,
            filename="musk.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 166)}
        self.converters["166"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="166", converters=self.converters)