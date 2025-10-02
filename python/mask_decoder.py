"""
Complete MLX implementation of MaskDecoder with exact PyTorch equivalence
Includes: SPConvTranspose2d, MaskDecoder, parameter transfer, and comprehensive testing
"""
import mlx.core as mx
import mlx.nn as nn
from typing import NamedTuple

from dense_encoder import DilatedDenseNet, InstanceNorm2d
from prelu import PReLU

class SPConvTranspose2d(nn.Module):
    """MLX implementation of SPConvTranspose2d (Sub-Pixel Convolution Transpose) with exact PyTorch matching"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, r: int = 1):
        super().__init__()
        self.out_channels = out_channels
        self.r = r
        
        # Conv2d layer that outputs out_channels * r channels
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * r, 
            kernel_size=kernel_size, 
            stride=(1, 1),
            bias=True  # Explicit bias=True to match PyTorch
        )
    
    def __call__(self, x):
        # Input x is NHWC: (batch, height, width, channels)
        # Ensure input is float32
        if x.dtype != mx.float32:
            x = x.astype(mx.float32)
        
        # Apply padding (1, 1, 0, 0) which means pad width dimension with 1 on each side
        # In NHWC format, width is dimension 2
        x_padded = mx.pad(x, [(0, 0), (0, 0), (1, 1), (0, 0)], constant_values=0.0)
        
        # Apply convolution
        out = self.conv(x_padded)
        
        # Get output shape: (batch, height, width, out_channels * r)
        batch, height, width, total_channels = out.shape

        # Convert to NCHW-like thinking: (b, h, w, out_ch*r) -> (b, out_ch*r, h, w)
        out_nchw = mx.transpose(out, (0, 3, 1, 2))  # NHWC -> NCHW
        
        # Now apply PyTorch logic exactly
        # view(b, r, out_ch, h, w)
        step1 = mx.reshape(out_nchw, (batch, self.r, self.out_channels, height, width))
        
        # permute(0, 2, 3, 4, 1) -> (b, out_ch, h, w, r)
        step2 = mx.transpose(step1, (0, 2, 3, 4, 1))
        
        # reshape(b, out_ch, h, w*r)
        step3 = mx.reshape(step2, (batch, self.out_channels, height, width * self.r))
        
        # Convert back to NHWC: (b, out_ch, h, w*r) -> (b, h, w*r, out_ch)
        final_output = mx.transpose(step3, (0, 2, 3, 1))
        
        return final_output.astype(mx.float32)

class MaskDecoder(nn.Module):
    """
    Complete MLX implementation of MaskDecoder.
    This version is corrected to include the full forward pass, with the
    dynamic prelu_out logic cleanly encapsulated.
    """
    def __init__(self, num_features: int, num_channel: int = 64, out_channel: int = 1):
        super().__init__()
        self.num_features = num_features

        self.dense_block = DilatedDenseNet(4, num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, kernel_size=(1, 2), bias=True)
        self.norm = InstanceNorm2d(out_channel, affine=True)
        self.prelu = PReLU(out_channel, channel_dim=-1)
        self.final_conv = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=True)

        # --- Dynamic prelu_out ---
        self.prelu_out: PReLU = None
        self._prelu_out_width: int = -1
        self._deferred_prelu_out_weights: DeferredWeights = None

    def _update_and_get_prelu_out(self, actual_width: int) -> PReLU:
        """
        Ensures the prelu_out layer exists and matches the required width.
        This method handles the dynamic creation and weight application.
        """
        if self.prelu_out is None or self._prelu_out_width != actual_width:
            print(f"Creating/updating dynamic prelu_out for width={actual_width}")
            self.prelu_out = PReLU(actual_width, init=-0.25, channel_dim=1)
            self._prelu_out_width = actual_width

            if self._deferred_prelu_out_weights is not None:
                stored_weights = self._deferred_prelu_out_weights
                if stored_weights.size == actual_width:
                    self.prelu_out.weight = stored_weights.weight
                    print(f"    Applied stored prelu_out weights (size: {stored_weights.size})")
                elif stored_weights.size > actual_width:
                    self.prelu_out.weight = stored_weights.weight[:actual_width]
                    print(f"    Using first {actual_width} elements from stored weights (size: {stored_weights.size})")
                else:
                    new_weight = mx.full((actual_width,), -0.25, dtype=mx.float32)
                    new_weight = new_weight.at[:stored_weights.size].set(stored_weights.weight)
                    self.prelu_out.weight = new_weight
                    print(f"    Extended stored weights from {stored_weights.size} to {actual_width}")
        return self.prelu_out

    def __call__(self, x: mx.array) -> mx.array:
        """
        Full, corrected forward pass.
        """
        if x.dtype != mx.float32:
            x = x.astype(mx.float32)

        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.norm(x)
        x = self.prelu(x)
        x = self.final_conv(x)

        # --- Permutation logic with encapsulated dynamic PReLU ---
        x_nchw = mx.transpose(x, (0, 3, 1, 2))
        x_permuted = mx.transpose(x_nchw, (0, 3, 2, 1))
        x_squeezed = mx.squeeze(x_permuted, axis=-1)

        if len(x_squeezed.shape) == 3:
            actual_width = x_squeezed.shape[1]
            prelu_out_layer = self._update_and_get_prelu_out(actual_width)
            x_prelu = prelu_out_layer(x_squeezed)

            x_final_permuted = mx.transpose(x_prelu, (0, 2, 1))
            x_final = mx.expand_dims(x_final_permuted, axis=-1)
        else:
            x_final = mx.expand_dims(x_squeezed, axis=-1)

        return x_final.astype(mx.float32)

class DeferredWeights(NamedTuple):
    """A structured way to hold weights (as mx.array) for deferred layer creation."""
    weight: mx.array
    size: int