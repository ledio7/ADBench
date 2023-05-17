from river import stream

from . import base


class Cardio(base.FileDataset):
    

    def __init__(self):
        super().__init__(
            n_samples=1_831,
            n_features=21,
            filename="cardio.csv",
            task=base.BINARY_CLF,
        )
        self.converters = {f"{i}": float for i in range(0, 21)}
        self.converters["21"] = float

    def __iter__(self):
        return stream.iter_csv(self.path, target="21", converters=self.converters)