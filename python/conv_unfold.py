import mlx.core as mx
import mlx.nn as nn
from typing import Union, Tuple

class ConvUnfold(nn.Module):
    """
    MLX implementation of a TFLite/CoreML-compatible `unfold` operation.

    This module uses a `mlx.nn.Conv2d` layer with a crafted, non-trainable
    identity-style kernel to perform the unfolding (patch extraction) operation.
    It is a direct translation of the PyTorch version.
    """
    def __init__(self,
                 in_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()

        # Normalize kernel_size, stride, and padding to tuples
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.in_channels = in_channels

        # The number of output channels for the convolution will be the
        # total number of elements in each unfolded block.
        patch_elements = self.kernel_size[0] * self.kernel_size[1]
        self.out_channels = self.in_channels * patch_elements

        # --- Create the non-trainable identity-style kernel ---
        # 1. Create a flat identity matrix. Shape: (patch_elements, patch_elements)
        identity_weight = mx.eye(patch_elements)

        # 2. Repeat for each input channel for grouped convolution.
        # Shape: (in_channels * patch_elements, patch_elements)
        identity_weight = mx.repeat(identity_weight, self.in_channels, axis=0)

        # 3. Reshape into the 4D kernel format required by MLX Conv2d (OHWI).
        # Shape: (out_channels, kH, kW, in_channels_per_group)
        # For grouped convolution, in_channels_per_group is 1.
        final_weight = mx.reshape(
            identity_weight,
            (self.out_channels, self.kernel_size[0], self.kernel_size[1], 1)
        )

        # Create the convolution layer
        self.unfold_conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels, # This makes it a depthwise-like operation
            bias=False
        )

        # 4. Set the crafted weight and make it non-trainable.
        # The layer is now "frozen" with the identity kernel.
        self.unfold_conv.weight = final_weight
        # Freezing parameters in MLX is done by not including them in `trainable_parameters()`
        # which is the default for attributes not created by `mlx.nn` layers.

    def __call__(self, x: mx.array) -> mx.array:
        """
        Applies the unfold operation.
        Args:
            x (mx.array): Input tensor in NHWC format.
        Returns:
            mx.array: Unfolded tensor. The output shape of Conv2d is (B, H_out, W_out, C_out),
                      which needs to be reshaped to match the standard unfold output.
        """
        # The output of the convolution will have the shape:
        # (B, H_out, W_out, C_out), where C_out is (patch_size * in_channels).
        conv_output = self.unfold_conv(x)

        B, H_out, W_out, C_out = conv_output.shape

        # Reshape to match the standard unfold output format:
        # (Batch, Num_Patches, Patch_Dimension)
        # Num_Patches = H_out * W_out
        # Patch_Dimension = C_out
        return mx.reshape(conv_output, (B, H_out * W_out, C_out))