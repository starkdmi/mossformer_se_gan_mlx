import mlx.core as mx
import mlx.nn as nn

class LayerNormalization4D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # Parameters: for NHWC format, shape should be (1, 1, 1, channels)
        self.gamma = mx.ones((channels,))  # Will broadcast to (1, 1, 1, channels)
        self.beta = mx.zeros((channels,))  # Will broadcast to (1, 1, 1, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # Input x is NHWC: (B, H, W, C)
        # PyTorch normalizes over channel dimension for each spatial location
        # This means: for each (B, H, W) position, normalize across C channels
        
        # Compute mean and variance over channel dimension (axis=3 in NHWC)
        mean = mx.mean(x, axis=3, keepdims=True)  # Shape: (B, H, W, 1)
        var = mx.var(x, axis=3, keepdims=True)    # Shape: (B, H, W, 1)
        
        # Normalize using rsqrt for better numerical stability
        x_norm = (x - mean) * mx.rsqrt(var + self.eps)
        
        # Apply learnable parameters (will broadcast correctly)
        return x_norm * self.gamma + self.beta

class LayerNormalization4DCF(nn.Module):
    def __init__(self, shape: tuple, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        channels, freqs = shape
        # Parameters: for NHWC format, shape should be (freqs, channels)
        # This will broadcast to (1, 1, freqs, channels) when applied to (B, H, W, C)
        self.gamma = mx.ones((freqs, channels))  # Shape: (freqs, channels)
        self.beta = mx.zeros((freqs, channels))  # Shape: (freqs, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # Input x is NHWC: (B, H, W, C) where W=freqs, C=channels
        # PyTorch LayerNormalization4DCF normalizes over the last two dimensions
        # which correspond to (W, C) in NHWC format = axes (2, 3)

        # Compute mean and variance over the last two axes (frequency and channel)
        mean = mx.mean(x, axis=(2, 3), keepdims=True)  # Shape: (B, H, 1, 1)
        var = mx.var(x, axis=(2, 3), keepdims=True)    # Shape: (B, H, 1, 1)
        
        # Normalize
        x_norm = (x - mean) * mx.rsqrt(var + self.eps)

        # Apply learnable parameters
        # gamma and beta have shape (freqs, channels) and will broadcast to (B, H, freqs, channels)
        return x_norm * self.gamma + self.beta
