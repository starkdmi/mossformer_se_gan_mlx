import mlx.core as mx
import mlx.nn as nn

class SELayer(nn.Module):
    """
    A direct and clean MLX translation of the provided PyTorch SELayer.
    This module expects NHWC input: (batch, height, width, channels).
    This version uses individual attributes to match the weight loading script.
    """
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        reduced_channel = channel // reduction

        # --- Average pooling path ---
        self.avg_pool_fc1 = nn.Linear(channel, reduced_channel, bias=True)
        self.avg_pool_relu = nn.ReLU()
        self.avg_pool_fc2 = nn.Linear(reduced_channel, channel, bias=True)
        self.avg_pool_sigmoid = nn.Sigmoid()

        # --- Max pooling path ---
        self.max_pool_fc1 = nn.Linear(channel, reduced_channel, bias=True)
        self.max_pool_relu = nn.ReLU()
        self.max_pool_fc2 = nn.Linear(reduced_channel, channel, bias=True)
        self.max_pool_sigmoid = nn.Sigmoid()

    def __call__(self, x: mx.array) -> mx.array:
        # The input tensor `x` has NHWC layout: (B, T, Q, C)
        
        # --- Squeeze ---
        # For NHWC input, squeeze over spatial dimensions (axes 1 and 2)
        x_avg_squeezed = mx.mean(x, axis=(1, 2))  # (B, C)
        x_max_squeezed = mx.max(mx.max(x, axis=2), axis=1)  # (B, C)

        # --- Excite (Average pool path) ---
        y_avg = self.avg_pool_fc1(x_avg_squeezed)
        y_avg = self.avg_pool_relu(y_avg)
        y_avg = self.avg_pool_fc2(y_avg)
        y_avg = self.avg_pool_sigmoid(y_avg)
        
        # --- Excite (Max pool path) ---
        y_max = self.max_pool_fc1(x_max_squeezed)
        y_max = self.max_pool_relu(y_max)
        y_max = self.max_pool_fc2(y_max)
        y_max = self.max_pool_sigmoid(y_max)

        # --- Scale ---
        # Reshape for broadcasting: (B, C) -> (B, 1, 1, C) for NHWC layout
        attention_weights = mx.expand_dims(mx.expand_dims(y_avg + y_max, axis=1), axis=2)
        
        # Scale the original input `x`
        return x * attention_weights