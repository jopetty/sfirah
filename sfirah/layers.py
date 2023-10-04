from torch import Tensor, nn


class AvgPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forwward(self, x: Tensor) -> Tensor:
        return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class IndexPool(nn.Module):
    def __init__(self, dim: int, index: int):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: Tensor) -> Tensor:
        return x.select(dim=self.dim, index=self.index)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, index={self.index}"


class SumPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.sum(dim=self.dim)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
