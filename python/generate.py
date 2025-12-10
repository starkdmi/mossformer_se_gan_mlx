"""
MossFormer GAN Speech Enhancement - MLX Implementation
Removes background noise from audio files using deep learning.

Usage:
    python generate.py --input audio.wav --output ./output --model fp32
"""
import os
import gc
import time
import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf
import yaml
import tqdm
from huggingface_hub import hf_hub_download

# --- NATIVE MLX Modules ---
from stft import stft, create_periodic_hann_window_mlx
from istft import istft

# --- Model ---
from model import MossFormer

# --- Constants ---
HF_REPO_ID = "starkdmi/MossFormer_GAN_SE_16K_MLX"
AVAILABLE_MODELS = {
    "fp32": "model_fp32.safetensors",
    "fp16": "model_fp16.safetensors",
    "int8": "model_int8.safetensors",
    "int4": "model_int4.safetensors"
}

# --- Utility Functions ---
def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_model_files(model_precision: str, cache_dir: str = None):
    """
    Download model weights and config from HuggingFace Hub.

    Args:
        model_precision: Model precision ('fp32', 'fp16', 'int8', 'int4')
        cache_dir: Optional cache directory for downloaded files

    Returns:
        Tuple of (config_path, weights_path)
    """
    if model_precision not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model precision. Choose from: {list(AVAILABLE_MODELS.keys())}")

    print(f"Downloading {model_precision} model from HuggingFace Hub...")

    # Download config
    config_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="config.yaml",
        cache_dir=cache_dir
    )

    # Download model weights
    weights_filename = AVAILABLE_MODELS[model_precision]
    weights_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=weights_filename,
        cache_dir=cache_dir
    )

    print(f"Model files downloaded successfully.")
    return config_path, weights_path

class AttrDict(argparse.Namespace):
    """Dictionary that allows attribute-style access."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getattr__(self, name):
        return self.__dict__.get(name, None)

def preprocess_audio(audio_path: str, target_sample_rate: int):
    """
    Load and preprocess audio file for speech enhancement.

    Args:
        audio_path: Path to input audio file
        target_sample_rate: Target sampling rate (typically 16000 Hz)

    Returns:
        MLX array of normalized audio samples
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Input audio file not found: {audio_path}")

    # Load audio
    audio, fs = sf.read(audio_path)

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if necessary
    if fs != target_sample_rate:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=fs, target_sr=target_sample_rate)
        except ImportError:
            raise ImportError("librosa is required for resampling. Install it with: pip install librosa")

    # Normalize audio
    audio = audio.astype('float32')
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio /= max_val

    return mx.array(audio).reshape(1, -1)

class MlxInferenceWrapper:
    """An MLX-native orchestrator for the enhancement pipeline with cached iSTFT."""
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.n_fft = int(args.fft_len)
        self.hop_length = int(args.win_inc)
        self.win_length = int(args.win_len)
        self.stft_window = create_periodic_hann_window_mlx(self.win_length)
        self.istft_window = self.stft_window
        self.use_binary_mask = getattr(args, 'use_binary_mask', False) # use True for 4-bit
        self._forward_compiled = mx.compile(self._forward)

    def _power_compress(self, real, imag):
        mag = mx.sqrt(real**2 + imag**2 + 1e-8)
        phase = mx.arctan2(imag, real)
        mag = mag ** 0.3
        real_compress = mag * mx.cos(phase)
        imag_compress = mag * mx.sin(phase)
        return mx.stack([real_compress, imag_compress], axis=1)

    def _power_uncompress(self, real, imag):
        mag = mx.sqrt(real**2 + imag**2 + 1e-8)
        phase = mx.arctan2(imag, real)
        mag = mag ** (1.0 / 0.3)
        real_uncompress = mag * mx.cos(phase)
        imag_uncompress = mag * mx.sin(phase)
        return mx.stack([real_uncompress, imag_uncompress], axis=-1)
    
    def _forward(self, inputs):
        input_len = inputs.shape[-1]
        norm_factor = mx.sqrt(input_len / (mx.sum(inputs**2.0, axis=-1, keepdims=True) + 1e-9))
        normed_inputs = inputs * norm_factor

        noisy_real, noisy_imag = stft(normed_inputs, self.n_fft, self.hop_length, self.win_length, self.stft_window)

        # STFT returns (B, F, T), we need to compress and prepare for model
        inputs_spec_compressed = self._power_compress(noisy_real, noisy_imag)
        # inputs_spec_compressed is now (B, 2, F, T), transpose to (B, 2, T, F) for model input
        inputs_spec_compressed = inputs_spec_compressed.transpose(0, 1, 3, 2)
        
        # This now calls the model
        out_list = self.model(inputs_spec_compressed)
        
        # Model outputs (B, 1, T, F), we need to convert to (B, F, T) for further processing
        pred_real_tf, pred_imag_tf = out_list[0].squeeze(1), out_list[1].squeeze(1)
        pred_real, pred_imag = pred_real_tf.transpose(0, 2, 1), pred_imag_tf.transpose(0, 2, 1)
        
        pred_spec_uncompressed = self._power_uncompress(pred_real, pred_imag)
        clean_real, clean_imag = pred_spec_uncompressed[..., 0], pred_spec_uncompressed[..., 1]
        
        noisy_mag = mx.sqrt(noisy_real**2 + noisy_imag**2 + 1e-9)
        clean_mag = mx.sqrt(clean_real**2 + clean_imag**2 + 1e-9)

        # Soft mask
        mask = mx.clip(clean_mag / (noisy_mag + 1e-9), a_min=0.0, a_max=1.0)
        if self.use_binary_mask:
            # Helps to remove vocals from background for 4-bit quantization
            mask = mx.where(mask > 0.5, 1.0, 0.0) # 0.5 threshold

        background_real, background_imag = noisy_real * (1 - mask), noisy_imag * (1 - mask)
        
        enhanced_audio = istft(
            clean_real, clean_imag, self.n_fft, self.hop_length, self.win_length, self.istft_window, True, input_len
        )
        background_audio = istft(
            background_real, background_imag, self.n_fft, self.hop_length, self.win_length, self.istft_window, True, input_len
        )
        enhanced_audio /= norm_factor
        background_audio /= norm_factor
        
        return enhanced_audio, background_audio

    def __call__(self, inputs: mx.array) -> dict:
        enhanced_audio, background_audio = self._forward_compiled(inputs)
        mx.eval(enhanced_audio, background_audio)
        return {'enhanced_audio': enhanced_audio, 'background_audio': background_audio}

def process_long_audio_by_chunking_mlx(full_audio_tensor, inference_runner, window_len, stride_len):
    padding = (window_len - stride_len) // 2
    original_len = full_audio_tensor.shape[-1]
    padded_audio = mx.pad(full_audio_tensor, ((0, 0), (padding, padding)), constant_values=0.0)
    total_padded_len = padded_audio.shape[-1]
    output_waveform_enhanced, output_waveform_background = mx.zeros_like(padded_audio), mx.zeros_like(padded_audio)
    window_sum = mx.zeros_like(padded_audio)
    stitching_window = mx.array(np.hanning(window_len))
    current_pos = 0
    pbar = tqdm.tqdm(total=total_padded_len, desc="Processing (MLX Pipeline)", unit="samples")
    while True:
        end_pos = current_pos + window_len
        if end_pos >= total_padded_len:
            last_chunk_start = total_padded_len - window_len
            chunk = padded_audio[:, last_chunk_start:]
            processed_chunks_dict = inference_runner(chunk)
            output_waveform_enhanced[:, last_chunk_start:] += processed_chunks_dict['enhanced_audio'] * stitching_window
            output_waveform_background[:, last_chunk_start:] += processed_chunks_dict['background_audio'] * stitching_window
            window_sum[:, last_chunk_start:] += stitching_window**2
            pbar.update(total_padded_len - pbar.n)
            break
        chunk = padded_audio[:, current_pos:end_pos]
        processed_chunks_dict = inference_runner(chunk)
        output_waveform_enhanced[:, current_pos:end_pos] += processed_chunks_dict['enhanced_audio'] * stitching_window
        output_waveform_background[:, current_pos:end_pos] += processed_chunks_dict['background_audio'] * stitching_window
        window_sum[:, current_pos:end_pos] += stitching_window**2
        pbar.update(stride_len)
        current_pos += stride_len
    pbar.close()
    window_sum = mx.maximum(window_sum, 1e-8)
    normalized_output_enhanced = output_waveform_enhanced / window_sum
    normalized_output_background = output_waveform_background / window_sum
    return (
        normalized_output_enhanced[:, padding : padding + original_len],
        normalized_output_background[:, padding : padding + original_len]
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MossFormer GAN Speech Enhancement - Remove background noise from audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input audio file (WAV, FLAC, MP3, etc.)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for enhanced audio files"
    )

    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(AVAILABLE_MODELS.keys()),
        default="fp32",
        help="Model precision (fp32=best quality, int4=fastest)"
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for downloaded models (default: HuggingFace cache)"
    )

    parser.add_argument(
        "--window-size",
        type=float,
        default=1.0,
        help="Processing window size in seconds"
    )

    parser.add_argument(
        "--stride",
        type=float,
        default=0.5,
        help="Processing stride in seconds (overlap = window_size - stride)"
    )

    return parser.parse_args()

def main():
    """Main entry point for speech enhancement."""
    # Parse command-line arguments
    cli_args = parse_arguments()

    # Create output directory
    output_dir = Path(cli_args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MossFormer GAN Speech Enhancement - MLX")
    print("=" * 70)

    # Download model files from HuggingFace Hub
    config_path, weights_path = download_model_files(
        model_precision=cli_args.model,
        cache_dir=cli_args.cache_dir
    )

    # Load configuration
    config = load_config(config_path)
    args = AttrDict(**config)

    # Preprocess input audio
    print(f"\nLoading audio: {cli_args.input}")
    audio_tensor = preprocess_audio(cli_args.input, args.sampling_rate)
    duration = audio_tensor.shape[-1] / args.sampling_rate
    print(f"Audio duration: {duration:.2f} seconds")

    # Initialize model
    print(f"\nInitializing {cli_args.model.upper()} model...")
    model = MossFormer(args)

    # Load model weights
    model.load_weights(weights_path)

    # Configure binary mask for 4-bit quantization
    if cli_args.model == "int4":
        setattr(args, "use_binary_mask", True)
        print("Using binary mask for 4-bit quantization")

    # Create inference wrapper
    inference_wrapper = MlxInferenceWrapper(model, args)

    # Calculate processing parameters
    window_len = int(cli_args.window_size * args.sampling_rate)
    stride_len = int(cli_args.stride * args.sampling_rate)

    # Process audio
    print(f"\nProcessing audio (window={cli_args.window_size}s, stride={cli_args.stride}s)...")

    # Clean run
    # gc.collect()
    # mx.clear_cache()
    # Benchmark begin
    mx.reset_peak_memory()
    start_time = time.time()

    enhanced_output, background_output = process_long_audio_by_chunking_mlx(
        audio_tensor, inference_wrapper, window_len, stride_len
    )

    mx.eval(enhanced_output, background_output)
    # Benchmark finish
    elapsed = time.time() - start_time
    peak_mem_bytes = mx.get_peak_memory()

    print(f"\nProcessing completed in {elapsed:.2f} seconds")
    print(f"Real-time factor: {duration / elapsed:.2f}x")
    print(f"Peak Memory: {peak_mem_bytes / (1024 * 1024):.2f} MB")

    # Save output files
    input_name = Path(cli_args.input).stem
    enhanced_path = output_dir / f"{input_name}_enhanced.wav"
    background_path = output_dir / f"{input_name}_background.wav"

    print(f"\nSaving enhanced audio to: {enhanced_path}")
    sf.write(str(enhanced_path), np.array(enhanced_output.squeeze()), args.sampling_rate)

    print(f"Saving background audio to: {background_path}")
    sf.write(str(background_path), np.array(background_output.squeeze()), args.sampling_rate)

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()