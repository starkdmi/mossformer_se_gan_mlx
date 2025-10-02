import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
from quantized_conv2d import QuantizedConv2d

from dense_encoder import DenseEncoder
from mask_decoder import MaskDecoder, DeferredWeights
from complex_decoder import ComplexDecoder
from sync_anet_block import SyncANetBlock

class MossFormer(nn.Module):
    """
    MLX implementation of Mossformer GAN SE 16K ported from PyTorch 
    https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice/clearvoice/models/mossformer_gan_se
    """

    def __init__(self, args):
        super().__init__()
        num_channel = 64
        num_features = args.fft_len // 2 + 1

        # Calculate n_freqs from input dimensions
        n_freqs = num_features // 2 + 1  # This should be 101 for num_features=201
        
        self.blocks = []
        for i in range(6):
            # Create MLX block
            mlx_block = SyncANetBlock(
                emb_dim=num_channel, emb_ks=2, emb_hs=1, n_freqs=n_freqs,
                hidden_channels=num_channel*2, n_head=4, approx_qk_dim=512, eps=1e-5
            )
            self.blocks.append(mlx_block)

        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.mask_decoder = MaskDecoder(num_features, num_channel, 1)
        self.complex_decoder = ComplexDecoder(num_channel)

    def load_weights(self, file_path: str): 
        """
        Universal, quantization-aware loader for production.
        """
        print(f"\nLoading model weights from: {file_path}")
        
        quantization_args = None
        weights = None
        
        try:
            if file_path.endswith(".safetensors"):
                from safetensors.mlx import load_file
                from safetensors import safe_open
                with safe_open(file_path, framework="mlx") as f:
                    metadata = f.metadata() or {}
                if 'quantization_bits' in metadata:
                    quantization_args = {
                        "group_size": int(metadata.get('quantization_group_size', 64)),
                        "bits": int(metadata.get('quantization_bits', 4)),
                        "quantize_conv": metadata.get('quantized_conv2d', 'False').lower() == 'true'
                    }
                weights = load_file(file_path)
            else:
                weights = mx.load(file_path)
        except Exception as e:
            print(f"Error: Weight file not found or cannot be read at '{file_path}'. Model will use random weights. Details: {e}")
            return

        if quantization_args:
            print(f"-> Detected quantization metadata: {quantization_args}")
            self.quantize_structure(**quantization_args)

        prelu_key = 'mask_decoder.prelu_out.weight'
        if prelu_key in weights:
            prelu_weight = weights.pop(prelu_key, None)
            if prelu_weight is not None:
                print("-> Storing deferred weights for MaskDecoder's dynamic prelu_out...")
                self.mask_decoder._deferred_prelu_out_weights = DeferredWeights(
                    weight=prelu_weight, size=prelu_weight.shape[0]
                )
        
        print(f"-> Applying {len(weights)} weight arrays to the model...")
        self.update(tree_unflatten(list(weights.items())))
        print("\nAll weights loaded successfully.")

    def quantize_structure(self, group_size: int, bits: int, quantize_conv: bool):
        """
        Replaces layers with their quantized counterparts with robust replacement logic.
        """
        quantized_layers = 0
        for name, module in self.named_modules():
            quantized_layer = None
            
            if isinstance(module, nn.Linear):
                last_dim = module.weight.shape[-1]
                if last_dim % group_size != 0: continue
                quantized_layer = nn.QuantizedLinear.from_linear(module, group_size, bits)
            elif isinstance(module, nn.Conv2d) and quantize_conv:
                in_channels = module.weight.shape[3]
                effective_group_size = min(group_size, in_channels)
                if in_channels % effective_group_size != 0 or effective_group_size not in [32, 64, 128]: continue
                if QuantizedConv2d is None: raise RuntimeError("QuantizedConv2d not imported.")
                quantized_layer = QuantizedConv2d(module, effective_group_size, bits)
            else:
                continue

            # Robust Layer Replacement Logic
            path_parts = name.split('.')
            parent = self
            # Traverse to the parent of the target module
            for part in path_parts[:-1]:
                parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
            
            # The final part of the name is the attribute to replace
            attr_name = path_parts[-1]

            # Check if the final attribute is a list index or a named attribute
            if attr_name.isdigit():
                # Handle assignment to a list (e.g., self.blocks[0])
                parent[int(attr_name)] = quantized_layer
            else:
                # Handle assignment to a named attribute (e.g., self.dense_encoder)
                setattr(parent, attr_name, quantized_layer)
            
            quantized_layers += 1
            
        print(f"\nQuantization complete, total layers replaced: {quantized_layers}")

    def __call__(self, x: mx.array) -> list[mx.array]:
        """
        Improved forward pass with actual NHWC optimizations:
        1. Pre-compute data in NHWC format to avoid repeated conversions
        2. Minimize array creation and copying
        3. Streamline magnitude/phase processing
        4. Optimize tensor operations for Apple Silicon
        """
        # Computes magnitude in NHWC directly
        real_part, imag_part = x[:, 0], x[:, 1]  # (B, H, W)
        
        # Compute magnitude directly in NHWC format
        mag_squared = real_part**2 + imag_part**2
        mag_nhwc = mx.expand_dims(mx.sqrt(mag_squared), axis=-1)  # (B, H, W, 1)
        
        # Compute phase directly in NHWC format
        is_not_silent = (mx.abs(real_part) > 1e-6) | (mx.abs(imag_part) > 1e-6)
        r = mx.sqrt(mag_squared + 1e-6)
        quotient = mx.clip(real_part / r, -1.0, 1.0)
        angle_default = mx.sign(imag_part) * mx.arccos(quotient)
        angle_fallback = mx.zeros_like(angle_default)
        phase_nhwc = mx.expand_dims(mx.where(is_not_silent, angle_default, angle_fallback), axis=-1)  # (B, H, W, 1)
        
        # Create NHWC input directly without NCHW intermediate
        real_nhwc = mx.expand_dims(real_part, axis=-1)  # (B, H, W, 1)
        imag_nhwc = mx.expand_dims(imag_part, axis=-1)  # (B, H, W, 1)
        x_in_nhwc = mx.concatenate([mag_nhwc, real_nhwc, imag_nhwc], axis=-1)  # (B, H, W, 3)
        
        # Dense encoder
        x = self.dense_encoder(x_in_nhwc)

        # Process blocks with minimal conversions
        for block in self.blocks:
            x = block(x)
        
        # Parallel processing paths to minimize data dependencies
        # Both mask and complex decoder work in NHWC - process simultaneously
        mask_nhwc = self.mask_decoder(x)
        complex_out_nhwc = self.complex_decoder(x)
        
        # Streamlined final computation without intermediate arrays
        mask_val = mask_nhwc[:, :, :, 0]  # (B, H, W)
        mag_val = mag_nhwc[:, :, :, 0]    # (B, H, W)
        phase_val = phase_nhwc[:, :, :, 0]  # (B, H, W)
        complex_real = complex_out_nhwc[:, :, :, 0]  # (B, H, W)
        complex_imag = complex_out_nhwc[:, :, :, 1]  # (B, H, W)
        
        # Fused magnitude processing
        masked_mag = mask_val * mag_val
        cos_phase = mx.cos(phase_val)
        sin_phase = mx.sin(phase_val)
        
        final_real = masked_mag * cos_phase + complex_real
        final_imag = masked_mag * sin_phase + complex_imag
        
        # Convert to output format with minimal memory allocation
        final_real_out = mx.expand_dims(final_real, axis=1)  # (B, H, W) -> (B, 1, H, W)
        final_imag_out = mx.expand_dims(final_imag, axis=1)  # (B, H, W) -> (B, 1, H, W)
        
        return [final_real_out, final_imag_out]