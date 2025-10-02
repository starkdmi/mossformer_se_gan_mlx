import mlx.core as mx
import mlx.nn as nn
from typing import Optional

class UniDeepFsmn(nn.Module):
    """
    MLX implementation of UniDeepFsmn, converted from PyTorch.

    This model performs a sequence modeling operation using linear projections
    and a temporal convolution to incorporate memory, followed by residual
    connections. It is designed to be a drop-in replacement for the PyTorch
    version when using the MLX framework.

    Key differences from the PyTorch implementation:
    - `mlx.nn.Conv1d` expects input in `(N, L, C)` (batch, length, channels) format,
      whereas `torch.nn.Conv1d` expects `(N, C, L)`. This simplifies the forward
      pass as we don't need to transpose the data before the convolution.
    - `mx.pad` is used for manual padding, with a shape-specific padding tuple
      to target the sequence (length) dimension.
    """
    def __init__(self, input_dim: int, output_dim: int, lorder: int, hidden_size: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lorder = lorder
        self.rorder = lorder  # In this model, left and right order are the same
        self.hidden_size = hidden_size if hidden_size is not None else output_dim

        # --- Layer Definitions ---
        self.linear = nn.Linear(input_dim, self.hidden_size)
        self.project = nn.Linear(self.hidden_size, output_dim, bias=False)

        kernel_size = self.lorder + self.rorder - 1

        # The convolution is defined to match the PyTorch version's parameters.
        # It's a depthwise convolution because groups == in_channels.
        # Padding is done manually in the forward pass.
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            groups=input_dim,
            bias=False
        )

    def __call__(self, input_tensor: mx.array) -> mx.array:
        """
        Forward pass for the UniDeepFsmn_MLX model.
        
        Args:
            input_tensor (mx.array): Input of shape (batch, time, channels).
        
        Returns:
            mx.array: Output of the same shape as the input.
        """
        # 1. Linear projections with ReLU activation.
        # input_tensor: (B, T, C_in)
        f1 = mx.maximum(self.linear(input_tensor), 0)  # ReLU
        p1 = self.project(f1)  # p1 shape: (B, T, C_out)

        # 2. Manual padding for the temporal convolution.
        # We pad the time/length dimension (axis=1) for the convolution.
        padding = (self.lorder - 1, self.rorder - 1)
        # Pad spec for mx.pad: [(dim0_pad), (dim1_pad), (dim2_pad)]
        x_padded = mx.pad(p1, [(0, 0), padding, (0, 0)])

        # 3. Temporal convolution.
        # MLX Conv1d operates directly on (N, L, C) format.
        conv_out = self.conv1(x_padded)  # conv_out shape: (B, T, C_out)

        # 4. First residual connection (with the pre-convolution tensor).
        out = p1 + conv_out

        # 5. Second residual connection (with the original input).
        # This works because the problem constraints ensure input_dim == output_dim.
        return input_tensor + out