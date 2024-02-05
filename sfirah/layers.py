"""Some useful pooling layers which aren't provided by PyTorch.

These modules are similar to the nn Pooling layers in PyTorch, but
are tailored for use in transformers.
"""

from torch import Tensor, nn


class AvgPool(nn.Module):
    """Averages over a specified dimesion."""

    def __init__(self, dim: int):
        """Initialize the AvgPool module.

        Args:
            dim (int): The dimension to average over.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        return x.mean(dim=self.dim)

    def extra_repr(self) -> str:
        """Return the extra representation."""
        return f"dim={self.dim}"


class IndexPool(nn.Module):
    """Selects a specific index from a specified dimension."""

    def __init__(self, dim: int, index: int | None):
        """Initialize the IndexPool module.

        Args:
            dim (int): The dimension to select from.
            index (int): The index to select.
        """
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x: Tensor, index: Tensor | None) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
            index (Tensor | None): The index to select, if not fixed.
        """
        if (index is None) == (self.index is None):
            if (index is None) and (self.index is None):
                raise ValueError(
                    "You must provide either a fixed index at init or "
                    "provide an index during the forward pass."
                )
            else:
                raise ValueError(
                    "You cannot provide both a fixed index at init and "
                    "an index during the forward pass."
                )

        if index is not None:
            return x.gather(dim=self.dim, index=index)
        else:
            return x.select(dim=self.dim, index=self.index)

    def extra_repr(self) -> str:
        """Return the extra representation."""
        return f"dim={self.dim}, index={self.index if self.index is not None else '*'}"


class SumPool(nn.Module):
    """Sums over a specified dimesion."""

    def __init__(self, dim: int):
        """Initialize the SumPool module.

        Args:
            dim (int): The dimension to sum over.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x (Tensor): The input tensor.
        """
        return x.sum(dim=self.dim)

    def extra_repr(self) -> str:
        """Return the extra representation."""
        return f"dim={self.dim}"
