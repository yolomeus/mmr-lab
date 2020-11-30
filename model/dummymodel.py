from torch import Tensor
from torch.nn import Linear, ReLU, Module, Embedding


class DummyModel(Module):
    """Dummy model for testing the MVSA Pipeline"""

    def __init__(self):
        super().__init__()
        self.embedding = Embedding(38667, 128)
        self.lin = Linear(128, 512)
        self.lin2 = Linear(50176, 512)
        self.lin_out = Linear(512, 3)

    def forward(self, inputs):
        images, tokens = inputs

        y = images.mean(dim=1)
        y = y.flatten(1)
        y = self.lin2(y)
        y = ReLU()(y)

        x = self.embedding(tokens)
        x = x.mean(dim=1)

        x = self.lin(x)
        x = ReLU()(x)
        x = x + y

        x = self.lin_out(x)
        return x
