from river import stream

from . import base


class Arrhythmia(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=452,
            n_features=274,
            filename="arrhythmia.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 274)}
        self.converters["274"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="274", converters=self.converters)