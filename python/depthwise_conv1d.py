"""
Depthwise 1D convolutions using custom Metal kernel.

Exports:
    - depthwise_conv1d(): Function for depthwise convolution
    - DepthwiseConv1d: Depthwise 1D convolution
    - Conv1dFast: Drop-in replacement for nn.Conv1d with automatic optimization
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import metal_kernel

_depthwise_conv1d_kernel = metal_kernel(
    name="custom_depthwise_conv1d",
    input_names=["inp", "weight", "params"],
    output_names=["out"],
    source="""
        const int chan = int(thread_position_in_grid.x);
        const int time = int(thread_position_in_grid.y);
        const int batch = int(thread_position_in_grid.z);

        const int B = params[0];
        const int L_in = params[1];
        const int C = params[2];
        const int K = params[3];
        const int pad = params[4];
        const int L_out = params[5];

        if (batch >= B || time >= L_out || chan >= C) {
            return;
        }

        const int out_index = ((batch * L_out) + time) * C + chan;
        const int weight_base = chan * K;

        T acc = T(0);
        for (int k = 0; k < K; ++k) {
            const int in_time = time + k - pad;
            if (in_time >= 0 && in_time < L_in) {
                const int in_index = ((batch * L_in) + in_time) * C + chan;
                acc += inp[in_index] * weight[weight_base + k];
            }
        }

        out[out_index] = acc;
    """,
)

@mx.compile
def depthwise_conv1d(
    x: mx.array,
    weight: mx.array,
    *,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    stream=None,
    threadgroup_size: int = 1,  # Allow tuning: 1, 8, 16, or 32
) -> mx.array:
    """
    Depthwise 1-D convolution with optimized Metal kernel.
    """

    # Handle both weight formats: (C, K, 1) and (C, 1, K)
    if weight.ndim == 3 and weight.shape[1] == 1 and weight.shape[2] > 1:
        weight = weight.transpose(0, 2, 1)

    if (
        stride == 1
        and x.ndim == 3
        and weight.ndim == 3
        and weight.shape[2] == 1
        and groups == x.shape[2]
        and weight.shape[0] == x.shape[2]
        and (x.dtype == mx.float32 or x.dtype == mx.float16)
    ):
        kernel_size = weight.shape[1]
        output_length = x.shape[1] + 2 * padding - kernel_size + 1

        if output_length > 0 and padding * 2 == kernel_size - 1:
            x_contig = mx.contiguous(x, allow_col_major=False, stream=stream)
            weight_contig = mx.contiguous(weight, allow_col_major=False, stream=stream)

            params = mx.array(
                [
                    x_contig.shape[0],
                    x_contig.shape[1],
                    x_contig.shape[2],
                    kernel_size,
                    padding,
                    output_length,
                ],
                dtype=mx.int32,
            )

            outputs = _depthwise_conv1d_kernel(
                inputs=[x_contig, weight_contig, params],
                template=[("T", x_contig.dtype)],
                grid=(x_contig.shape[2], output_length, x_contig.shape[0]),
                threadgroup=(threadgroup_size, 1, 1),
                output_shapes=[[x_contig.shape[0], output_length, x_contig.shape[2]]],
                output_dtypes=[x_contig.dtype],
                stream=stream,
            )

            return outputs[0]

    return mx.conv1d(
        x,
        weight,
        stride=stride,
        padding=padding,
        groups=groups,
        stream=stream,
    )


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution Module using optimized Metal kernel."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = mx.zeros((channels, kernel_size, 1))
        self.bias = mx.zeros((channels,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        output = depthwise_conv1d(
            x, self.weight, stride=self.stride, padding=self.padding, groups=self.channels
        )
        if self.bias is not None:
            output = output + self.bias
        return output

class Conv1dFast(nn.Module):
    """
    Drop-in replacement for nn.Conv1d with optimized depthwise convolution.

    Automatically uses custom Metal kernel when groups == in_channels (depthwise).
    Falls back to standard nn.Conv1d for other configurations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        # Check if this is depthwise convolution
        self.is_depthwise = (
            groups == in_channels and in_channels == out_channels
        )

        if self.is_depthwise:
            # Use optimized kernel for depthwise
            self.weight = mx.zeros((in_channels, kernel_size, 1))
            self.bias = mx.zeros((in_channels,)) if bias else None
        else:
            # Fall back to standard nn.Conv1d
            self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size, stride, padding, bias
            )

    def __call__(self, x: mx.array) -> mx.array:
        if self.is_depthwise:
            output = depthwise_conv1d(
                x,
                self.weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
                threadgroup_size=32
            )
            if self.bias is not None:
                output = output + self.bias
            return output
        else:
            return self.conv(x)
