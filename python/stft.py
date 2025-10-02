import mlx.core as mx

def create_periodic_hann_window_mlx(win_length: int) -> mx.array:
    """
    Creates a periodic Hann window in pure MLX. Mathematically identical to torch.hann_window(periodic=True).
    """
    return 0.5 * (1 - mx.cos(2 * mx.pi * mx.arange(win_length) / win_length))

def stft(x: mx.array, n_fft: int, hop_length: int, win_length: int, 
                       window: mx.array, center: bool = True):
    """
    STFT with optimizations:
    - Minimal memory allocations
    - Vectorized operations
    - Optimized for repeated calls
    """
    batch_size, signal_len = x.shape
    
    # 1. Efficient padding using slice operations
    if center:
        pad_amount = n_fft // 2
        # Direct slice-based reflection (fastest approach)
        x_padded = mx.concatenate([
            x[:, 1:pad_amount + 1][:, ::-1],  # Left reflection
            x,                                 # Original signal
            x[:, -pad_amount - 1:-1][:, ::-1]  # Right reflection
        ], axis=-1)
        padded_len = signal_len + 2 * pad_amount
    else:
        x_padded = x
        padded_len = signal_len

    # 2. Single-shot framing and windowing
    num_frames = (padded_len - win_length) // hop_length + 1
    
    # Create frames with strides
    frames = mx.as_strided(
        x_padded, 
        shape=(batch_size, num_frames, win_length), 
        strides=(padded_len, hop_length, 1)
    )
    
    # 3. Apply window and handle FFT size in one operation
    if win_length == n_fft:
        # Perfect case - no padding needed
        windowed_frames = frames * window
    else:
        # Pad window to n_fft size and apply
        window_padded = mx.concatenate([window, mx.zeros(n_fft - win_length)])
        windowed_frames = frames * window[:win_length]  # Apply original window
        # Pad frames to n_fft
        windowed_frames = mx.concatenate([
            windowed_frames, 
            mx.zeros((batch_size, num_frames, n_fft - win_length))
        ], axis=-1)

    # 4. FFT with immediate transpose for memory efficiency
    stft_complex = mx.fft.rfft(windowed_frames, n=n_fft, axis=-1).transpose(0, 2, 1)

    return mx.real(stft_complex), mx.imag(stft_complex)