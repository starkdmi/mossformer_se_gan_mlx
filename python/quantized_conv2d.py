import mlx.core as mx
import mlx.nn as nn

class QuantizedConv2d(nn.Module):
    """
    A self-contained module for 2D convolution with quantized weights.
    This is a drop-in replacement for `mlx.nn.Conv2d` for quantization purposes.

    It robustly handles quantization by:
    1.  Receiving the original `nn.Conv2d` layer to correctly inherit all
        convolution parameters (stride, padding, etc.).
    2.  Determining a safe, supported group size for quantization.
    3.  Reshaping the 4D weight tensor to 2D before quantizing to use MLX's
        most stable quantization path.
    4.  Reshaping the dequantized weight back to 4D during the forward pass.
    5.  Correctly applying the bias term after the convolution operation.
    """
    def __init__(self, original_conv: nn.Conv2d, group_size: int, bits: int):
        super().__init__()
        
        # Store all original convolution parameters
        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.original_shape = original_conv.weight.shape
        
        # Store quantization parameters used for this specific layer
        self.bits = bits
        in_channels = self.original_shape[3]
        self.group_size = min(group_size, in_channels)
        
        # Reshape to 2D for robust quantization
        weight_reshaped = original_conv.weight.reshape(-1, in_channels)
        
        # Quantize using the effective group size
        w_quantized, w_scales, w_biases = mx.quantize(
            weight_reshaped, self.group_size, self.bits
        )
        
        self.weight = w_quantized.astype(mx.uint32)
        self.weight_scales = w_scales
        self.weight_biases = w_biases
        
        self.bias = original_conv.bias if "bias" in original_conv else None

    def __call__(self, x: mx.array) -> mx.array:
        # Dequantize the 2D matrix using the layer's specific parameters
        dequantized_2d = mx.dequantize(
            self.weight.astype(mx.uint32), 
            self.weight_scales, 
            self.weight_biases,
            self.group_size,
            self.bits
        ).astype(x.dtype)
        
        dequantized_weight = dequantized_2d.reshape(self.original_shape)
        
        output = mx.conv2d(
            x, dequantized_weight, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation,
            groups=self.groups
        )
        
        if self.bias is not None:
            output = output + self.bias
        return output