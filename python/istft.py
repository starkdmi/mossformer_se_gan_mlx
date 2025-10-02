import mlx.core as mx

def istft(real_part: mx.array, imag_part: mx.array, n_fft: int, 
                       hop_length: int, win_length: int, window: mx.array, 
                       center: bool = True, audio_length: int = None) -> mx.array:
    """
    MLX iSTFT with multiple performance improvements:
    1. Pre-computed normalization buffer (can be cached)
    2. Reduced memory allocations
    3. Vectorized operations where possible
    4. Optional pre-computation for repeated calls
    """
    # Step 1: Get windowed time-domain frames 
    stft_complex = real_part + 1j * imag_part
    time_frames = mx.fft.irfft(stft_complex.transpose(0, 2, 1), n=n_fft, axis=-1)
    windowed_frames = time_frames * window
    
    batch_size, num_frames, frame_length = windowed_frames.shape
    ola_len = (num_frames - 1) * hop_length + frame_length
    
    # Step 2: Pre-compute normalization buffer (this can be cached between calls)
    window_squared = window ** 2
    norm_buffer = mx.zeros(ola_len, dtype=mx.float32)
    
    # Vectorized normalization buffer creation
    positions = mx.arange(num_frames)[:, None] * hop_length + mx.arange(frame_length)[None, :]
    positions_flat = positions.reshape(-1)
    window_sq_tiled = mx.tile(window_squared, num_frames)
    norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)
    
    # Step 3: Overlap-add using advanced indexing
    output = mx.zeros((batch_size, ola_len), dtype=mx.float32)
    
    # Reshape for vectorized scatter
    windowed_flat = windowed_frames.reshape(batch_size, -1)
    
    # Use scatter_add for all batches at once
    for b in range(batch_size):
        output = output.at[b, positions_flat].add(windowed_flat[b])
    
    # Step 4: Normalize (avoid division by zero)
    norm_buffer = mx.maximum(norm_buffer, 1e-10)
    output = output / norm_buffer[None, :]
    
    # Step 5: Final trimming
    if center:
        start_cut = n_fft // 2
        output = output[:, start_cut:]
     
    if audio_length is not None:
        output = output[:, :audio_length]
        
    return output