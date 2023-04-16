from river import stream

from . import base


class Satimage(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=5_803,
            n_features=36,
            filename="satimage-2.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 36)}
        self.converters["y"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="y", converters=self.converters)