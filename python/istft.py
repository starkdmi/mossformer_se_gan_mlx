import mlx.core as mx

@mx.compile
def istft(real_part: mx.array, imag_part: mx.array, n_fft: int, 
          hop_length: int, win_length: int, window: mx.array, 
          center: bool = True, audio_length: int = None) -> mx.array:
    # Robust Window Padding (Safety Check)
    # Ensure window matches n_fft size (critical for correct broadcasting)
    if window.shape[0] < n_fft:
        pad = n_fft - window.shape[0]
        window = mx.concatenate([window, mx.zeros((pad,), dtype=window.dtype)])

    # Inverse FFT
    stft_complex = real_part + 1j * imag_part
    time_frames = mx.fft.irfft(stft_complex.transpose(0, 2, 1), n=n_fft, axis=-1)
    
    # Apply synthesis window
    windowed_frames = time_frames * window
    
    batch_size, num_frames, frame_length = windowed_frames.shape
    ola_len = (num_frames - 1) * hop_length + frame_length
    
    # Vectorized Overlap-Add (The Speedup)
    # Allocate flat output buffer (Batch * Time)
    output = mx.zeros((batch_size * ola_len), dtype=mx.float32)
    
    # Calculate indices for a SINGLE frame sequence
    positions = mx.arange(num_frames)[:, None] * hop_length + mx.arange(frame_length)[None, :]
    positions_flat = positions.reshape(-1)
    
    # Calculate global indices for ALL batches at once
    # We offset each batch's indices by (Batch_Index * OLA_Len)
    batch_offsets = mx.arange(batch_size) * ola_len
    global_indices = positions_flat[None, :] + batch_offsets[:, None]
    
    # Single scatter_add operation for the entire batch
    output = output.at[global_indices.reshape(-1)].add(windowed_frames.reshape(-1))
    
    # Reshape back to (Batch, Time)
    output = output.reshape(batch_size, ola_len)
    
    # Normalization
    # We compute the window sum buffer to normalize overlapping windows
    window_squared = window ** 2
    norm_buffer = mx.zeros(ola_len, dtype=mx.float32)
    window_sq_tiled = mx.tile(window_squared, num_frames)
    norm_buffer = norm_buffer.at[positions_flat].add(window_sq_tiled)
    
    # Avoid division by zero
    norm_buffer = mx.maximum(norm_buffer, 1e-10)
    output = output / norm_buffer[None, :]
    
    # Final Trimming
    if center:
        start_cut = n_fft // 2
        output = output[:, start_cut:]
     
    if audio_length is not None:
        output = output[:, :audio_length]

    return output
