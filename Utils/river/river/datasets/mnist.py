from river import stream

from . import base


class Mnist(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=7_603,
            n_features=100,
            filename="mnist.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 100)}
        self.converters["100"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="100", converters=self.converters)