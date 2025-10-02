import mlx.core as mx
import mlx.nn as nn

# Set a tolerance for floating point comparisons
ATOL = 1e-6

class DepthwiseConv1d(nn.Module):
    """
    A robust MLX implementation of 1D depthwise convolution using `mx.vmap`,
    reverted from the original user-provided file. This avoids using the
    problematic `groups` parameter in `nn.Conv1d`.
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        # PyTorch weight shape (C_out, C_in/groups, K) -> (C, 1, K)
        # This implementation correctly uses the (C, 1, K) shape.
        self.weight = mx.zeros((channels, 1, kernel_size))
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

        def single_channel_conv(x_ch, w_ch):
            # x_ch arrives as (Batch, Length) -> reshape to (Batch, Length, 1)
            x_in = mx.expand_dims(x_ch, axis=-1)
            # w_ch arrives as (1, Kernel) -> reshape to (1, Kernel, 1) for (O,K,I)
            w_in = mx.expand_dims(w_ch.T, axis=0)
            return mx.conv1d(x_in, w_in, stride=stride, padding=padding)

        # vmap over the channel dimension of the input and the filter dimension of the weight
        self.vmapped_conv = mx.vmap(single_channel_conv, in_axes=(2, 0), out_axes=0)

    def __call__(self, x: mx.array) -> mx.array:
        """Input x shape: (Batch, Length, Channels)"""
        # self.vmapped_conv output shape is (C, B, L_out, 1)
        output = self.vmapped_conv(x, self.weight)
        # Reshape from (C, B, L_out, 1) to (B, L_out, C)
        output = mx.squeeze(output, axis=-1).transpose(1, 2, 0)
        if self.bias is not None:
            output = output + self.bias
        return output


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