from river import stream

from . import base


class Breastw(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=683,
            n_features=9,
            filename="breastw.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 9)}
        self.converters["9"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="9", converters=self.converters)