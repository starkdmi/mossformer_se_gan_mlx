import mlx.core as mx
import mlx.nn as nn

from ffconvm import FFConvM
from fsmn import UniDeepFsmn
from mossformer import MossFormer
from se_layer import SELayer
from conv_unfold import ConvUnfold
from norms import LayerNormalization4D, LayerNormalization4DCF

class SyncANetBlock(nn.Module):
    def __init__(self, emb_dim: int, emb_ks: int, emb_hs: int, n_freqs: int,
                 hidden_channels: int, n_head: int = 4, approx_qk_dim: int = 512,
                 eps: float = 1e-5):
        super().__init__()
        self.emb_dim, self.emb_ks, self.emb_hs, self.n_head = emb_dim, emb_ks, emb_hs, n_head
        in_channels = emb_dim * emb_ks

        self.Fconv = nn.Conv2d(emb_dim, in_channels, (1, emb_ks), 1, groups=emb_dim)
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)
        self.intra_to_u = FFConvM(in_channels, hidden_channels)
        self.intra_to_v = FFConvM(in_channels, hidden_channels)
        self.intra_rnn = nn.Sequential(*[UniDeepFsmn(in_channels, hidden_channels, 20, hidden_channels) for _ in range(1)])
        self.intra_mossformer = MossFormer(dim=emb_dim, group_size=n_freqs)
        self.intra_linear = nn.ConvTranspose1d(hidden_channels, emb_dim, emb_ks, emb_hs)
        self.intra_se = SELayer(emb_dim, 1)

        self.inter_to_u = FFConvM(in_channels, hidden_channels)
        self.inter_to_v = FFConvM(in_channels, hidden_channels)
        self.inter_rnn = nn.Sequential(*[UniDeepFsmn(in_channels, hidden_channels, 20, hidden_channels) for _ in range(1)])
        self.inter_mossformer = MossFormer(dim=emb_dim, group_size=256)
        self.inter_linear = nn.ConvTranspose1d(hidden_channels, emb_dim, emb_ks, emb_hs)
        self.inter_se = SELayer(emb_dim, 1)

        self.conv_unfolder = ConvUnfold(kernel_size=(emb_ks, 1), stride=(emb_hs, 1), in_channels=emb_dim)

        E = mx.ceil(approx_qk_dim / n_freqs)
        assert emb_dim % n_head == 0
        
        self.attn_conv_Q, self.attn_conv_K, self.attn_conv_V = [], [], []
        for ii in range(n_head):
            self.attn_conv_Q.append(nn.Sequential(
                nn.Conv2d(emb_dim, E, 1), 
                nn.PReLU(num_parameters=1),
                LayerNormalization4DCF((E, n_freqs), eps=eps)  # (E, n_freqs) for Q and K
            ))
            self.attn_conv_K.append(nn.Sequential(
                nn.Conv2d(emb_dim, E, 1), 
                nn.PReLU(num_parameters=1),
                LayerNormalization4DCF((E, n_freqs), eps=eps)  # (E, n_freqs) for Q and K
            ))
            self.attn_conv_V.append(nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim // n_head, 1), 
                nn.PReLU(num_parameters=1),
                LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps)  # (emb_dim//n_head, n_freqs) for V
            ))

        self.attn_concat_proj = nn.Sequential(
            nn.Conv2d(emb_dim, emb_dim, 1), 
            nn.PReLU(num_parameters=1),
            LayerNormalization4DCF((emb_dim, n_freqs), eps=eps)
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.clip(x, a_min=-100.0, a_max=100.0)

        B, old_T, old_Q, C = x.shape
        T_pad = ((old_T - self.emb_ks + self.emb_hs - 1) // self.emb_hs) * self.emb_hs + self.emb_ks
        Q_pad = ((old_Q - self.emb_ks + self.emb_hs - 1) // self.emb_hs) * self.emb_hs + self.emb_ks

        x_padded = mx.pad(x, [(0, 0), (0, T_pad - old_T), (0, Q_pad - old_Q), (0, 0)])
        
        input_intra = x_padded
        x_intra = self.intra_norm(input_intra)

        x_intra = self.Fconv(x_intra)

        B, T, Q_out, C_out = x_intra.shape

        # Reshape for sequence processing: (B*T, Q_out, C_out)
        x_intra_reshaped = mx.reshape(x_intra, (B * T, Q_out, C_out))

        u = self.intra_to_u(x_intra_reshaped)
        v = self.intra_to_v(x_intra_reshaped)
        u = self.intra_rnn.layers[0](u)
        x_intra_proc = v * u

         # ConvTranspose1d - keep in channels-last format
        x_intra_proc = self.intra_linear(x_intra_proc)
        
        # Reshape back to 4D NHWC
        # The output of the layer is also channels-last, so no transpose is needed before reshape.
        x_intra_4d = mx.reshape(x_intra_proc, (B, T, -1, self.emb_dim))

        x_intra_4d = self.intra_mossformer(x_intra_4d)

        x_intra_se_out = self.intra_se(x_intra_4d)
        output_intra = x_intra_se_out + input_intra

        input_inter = output_intra
        x_inter = self.inter_norm(input_inter)
        x_inter_transposed = x_inter.transpose(0, 2, 1, 3)
        B_i, Q_i, T_i, C_i = x_inter_transposed.shape
        x_inter_reshaped = mx.reshape(x_inter_transposed, (B_i * Q_i, T_i, C_i))

        x_inter_unfolded = self.conv_unfolder(mx.expand_dims(x_inter_reshaped, axis=2))

        u = self.inter_to_u(x_inter_unfolded)
        v = self.inter_to_v(x_inter_unfolded)
        u = self.inter_rnn.layers[0](u)
        x_inter_proc = v * u
        
        # --- Call ConvTranspose1d directly on the CHANNELS-LAST tensor (inter block) ---
        x_inter_proc = self.inter_linear(x_inter_proc)
        
        # Reshape back to 4D
        B_in, _, Q_in, _ = input_inter.shape
        x_inter_4d = mx.reshape(x_inter_proc, (B_in, Q_in, -1, self.emb_dim))
        
        x_inter_4d = self.inter_mossformer(x_inter_4d)
        # Conditionally squeeze the batch dimension only when batch size is 1
        if x_inter_4d.shape[0] == 1:
            x_inter_3d = mx.squeeze(x_inter_4d, axis=0)
            # Transpose from (Q, T) -> (T, Q) to match input_inter format
            inter_rnn = mx.transpose(x_inter_3d, (1, 0, 2))  # (Q, T, C) -> (T, Q, C)
            inter_rnn = mx.expand_dims(inter_rnn, axis=0)
        else:
            # For batch size > 1, transpose from (B, Q, T, C) -> (B, T, Q, C)
            inter_rnn = mx.transpose(x_inter_4d, (0, 2, 1, 3))  # (B, Q, T, C) -> (B, T, Q, C)

        # SE layer
        x_inter_se_out = self.inter_se(inter_rnn)

        output_inter = x_inter_se_out + input_inter

        batch = output_inter[:, :old_T, :old_Q, :]

        # Attention computation to match PyTorch exactly
        all_Q, all_K, all_V = [], [], []
        for ii in range(self.n_head):
            # Each conv expects NHWC input and produces NHWC output
            q_out = self.attn_conv_Q[ii](batch)  # NHWC -> NHWC
            k_out = self.attn_conv_K[ii](batch)  # NHWC -> NHWC  
            v_out = self.attn_conv_V[ii](batch)  # NHWC -> NHWC
            
            all_Q.append(q_out)
            all_K.append(k_out)
            all_V.append(v_out)

        # Concatenate along batch dimension for multi-head processing
        Q = mx.concatenate(all_Q, axis=0)  # (n_head*B, T, Q, E)
        K = mx.concatenate(all_K, axis=0)  # (n_head*B, T, Q, E)
        V = mx.concatenate(all_V, axis=0)  # (n_head*B, T, Q, C//n_head)
        
        # Match PyTorch exactly: Q.transpose(1, 2).flatten(2), K.transpose(1, 2).flatten(2)
        # PyTorch starts with (B*n_head, C, T, Q) and does .transpose(1,2) -> (B*n_head, T, C, Q), then .flatten(2) -> (B*n_head, T, C*Q)
        # MLX starts with (B*n_head, T, Q, C) (NHWC), so to get (B*n_head, T, C*Q) we need:
        # (B*n_head, T, Q, C) -> transpose(2,3) -> (B*n_head, T, C, Q) -> flatten(2) -> (B*n_head, T, C*Q)
        Q_transposed = Q.transpose(0, 1, 3, 2)  # (n_head*B, T, Q, E) -> (n_head*B, T, E, Q)
        K_transposed = K.transpose(0, 1, 3, 2)  # (n_head*B, T, Q, E) -> (n_head*B, T, E, Q)
        Q_r = Q_transposed.reshape(Q.shape[0], old_T, -1)  # (n_head*B, T, E*Q)
        K_r = K_transposed.reshape(K.shape[0], old_T, -1)  # (n_head*B, T, E*Q)
        
        # For V: PyTorch does V.transpose(1, 2) then keeps shape info, then V.flatten(2)
        # PyTorch V starts as (B*n_head, C, T, Q) -> .transpose(1,2) -> (B*n_head, T, C, Q)
        # MLX V starts as (B*n_head, T, Q, C) -> to match we need (B*n_head, T, C, Q)
        V_transposed = V.transpose(0, 1, 3, 2)  # (n_head*B, T, Q, C//n_head) -> (n_head*B, T, C//n_head, Q)
        old_shape = V_transposed.shape  # Save shape for later reshape
        emb_dim = Q_r.shape[-1]  # This should match PyTorch's emb_dim
        V_r = V_transposed.reshape(V.shape[0], old_T, -1)  # (n_head*B, T, (C//n_head)*Q)

        # Attention computation: Q @ K.T -> (n_head*B, T, E*Q) @ (n_head*B, E*Q, T) -> (n_head*B, T, T)
        attn_mat = mx.matmul(Q_r, K_r.transpose(0, 2, 1)) / mx.sqrt(emb_dim)
        attn_mat = mx.clip(attn_mat, a_min=-50.0, a_max=50.0)
        attn_mat = nn.softmax(attn_mat, axis=2)

        # Apply attention to values: attn @ V -> (n_head*B, T, T) @ (n_head*B, T, (C//n_head)*Q) -> (n_head*B, T, (C//n_head)*Q)
        V_out = mx.matmul(attn_mat, V_r)
        
        # Reshape back to original shape, then transpose back like PyTorch
        V_out = V_out.reshape(old_shape)  # Back to (n_head*B, T, C//n_head, Q)
        V_out = V_out.transpose(0, 2, 1, 3)  # (n_head*B, T, C//n_head, Q) -> (n_head*B, C//n_head, T, Q) to match PyTorch
        
        # Match PyTorch final reshape: V.view(n_head, B, emb_dim, old_T, -1).transpose(0, 1).contiguous().view(B, n_head * emb_dim, old_T, -1)
        # V_out is now (n_head*B, C//n_head, T, Q) format matching PyTorch
        emb_dim_v = V_out.shape[1]  # C//n_head
        batch_reshaped = V_out.reshape(self.n_head, B, emb_dim_v, old_T, old_Q)  # (n_head, B, C//n_head, T, Q)
        batch_transposed = batch_reshaped.transpose(1, 0, 2, 3, 4)  # (B, n_head, C//n_head, T, Q)
        batch_final = batch_transposed.reshape(B, self.n_head * emb_dim_v, old_T, old_Q)  # (B, C, T, Q)
        
        # Convert from NCHW to NHWC for our pipeline
        batch_nhwc = batch_final.transpose(0, 2, 3, 1)  # (B, C, T, Q) -> (B, T, Q, C)
        
        # Final projection - expects NHWC
        batch = self.attn_concat_proj(batch_nhwc)
        
        # Final residual connection
        output = batch + output_inter[:, :old_T, :old_Q, :]

        return output