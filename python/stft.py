import mlx.core as mx

@mx.compile
def create_periodic_hann_window_mlx(win_length: int) -> mx.array:
    """
    Creates a periodic Hann window in pure MLX. Mathematically identical to torch.hann_window(periodic=True).
    """
    return 0.5 * (1 - mx.cos(2 * mx.pi * mx.arange(win_length) / win_length))

@mx.compile
def stft(x: mx.array, n_fft: int, hop_length: int, win_length: int, 
         window: mx.array, center: bool = True):
    # Ensure 2D input (Batch, Time)
    if x.ndim == 1:
        x = x[None, :]

    batch_size, signal_len = x.shape
    
    # Pad Window to n_fft (The "Masking" Trick)
    # Instead of padding every frame with zeros later (expensive), 
    # we pad the window once. The zeros in the window will "mask" the extra data.
    if win_length < n_fft:
        pad = n_fft - win_length
        window = mx.concatenate([window, mx.zeros((pad,), dtype=window.dtype)])
    
    # Pad Signal (Reflection Padding)
    if center:
        pad_amount = n_fft // 2
        x_padded = mx.concatenate([
            x[:, 1:pad_amount + 1][:, ::-1],  # Left reflection
            x,                                 # Original signal
            x[:, -pad_amount - 1:-1][:, ::-1]  # Right reflection
        ], axis=-1)
    else:
        x_padded = x
    
    # Calculate Exact Target Frames
    padded_len = x_padded.shape[-1]
    num_frames = (padded_len - win_length) // hop_length + 1
    
    # Ensure Signal is long enough for the stride
    # Since we are using a larger 'n_fft' stride, we might read past the end of the 
    # original data. We add dummy padding to the signal to prevent this. 
    # This extra data is garbage, but it gets multiplied by the window's zeros, so it's safe.
    target_len = (num_frames - 1) * hop_length + n_fft
    
    if padded_len < target_len:
        pad_extra = target_len - padded_len
        x_padded = mx.concatenate([x_padded, mx.zeros((batch_size, pad_extra), dtype=x.dtype)], axis=-1)
    
    # Create Zero-Copy Views (as_strided)
    shape = (batch_size, num_frames, n_fft)
    strides = (x_padded.shape[-1], hop_length, 1)
    
    frames = mx.as_strided(x_padded, shape=shape, strides=strides)
    
    # Broadcasting handles the batch dimension automatically
    stft_complex = mx.fft.rfft(frames * window, n=n_fft, axis=-1)
    
    # Transpose to (Batch, Freq, Time) and return
    stft_complex = stft_complex.transpose(0, 2, 1)

    return mx.real(stft_complex), mx.imag(stft_complex)
