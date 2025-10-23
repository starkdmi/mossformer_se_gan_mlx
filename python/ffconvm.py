import mlx.core as mx
import mlx.nn as nn

from depthwise_conv1d import DepthwiseConv1d

ATOL = 1e-6

class ConvModule(nn.Module):
    """MLX implementation of the simple ConvModule used in FFConvM."""

    def __init__(self, in_channels: int, kernel_size: int = 31):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = DepthwiseConv1d(
            channels=in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        """Input shape: (Batch, Length, Channels)"""
        return inputs + self.depthwise_conv(inputs)


class FFConvM(nn.Module):
    """The final MLX conversion of the PyTorch FFConvM module."""
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim_in)
        self.linear = nn.Linear(dim_in, dim_out)
        self.conv_module = ConvModule(dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        """Input shape: (Batch, Length, Channels)"""
        x = self.norm(x)
        x = self.linear(x)
        x = nn.silu(x)
        x = self.conv_module(x)
        return x

    def forward_detailed(self, x: mx.array):
        """A forward pass that returns intermediate outputs for verification."""
        x_norm = self.norm(x)
        x_linear = self.linear(x_norm)
        x_silu = nn.silu(x_linear)
        x_final = self.conv_module(x_silu)
        return x_norm, x_linear, x_silu, x_final