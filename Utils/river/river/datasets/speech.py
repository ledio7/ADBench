from river import stream

from . import base


class Speech(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=3_686,
            n_features=400,
            filename="speech.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 400)}
        self.converters["400"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="400", converters=self.converters)