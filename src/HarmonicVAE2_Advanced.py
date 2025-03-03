import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from typing import List, Optional, Tuple, Dict, Union

# --- HarmonicLayer ---
class HarmonicLayer(nn.Module):
    """
    A custom layer that incorporates harmonic relationships between frequencies, with optional dynamic learning.
    """
    def __init__(self, in_features, out_features, harmonic_weight=0.5, 
                 sample_rate=22050, n_mels=128, is_mel_scale=True, dynamic_harmonics=True):
        super(HarmonicLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_mel_scale = is_mel_scale
        self.dynamic_harmonics = dynamic_harmonics
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Harmonic matrix based on music theory intervals
        harmonic_matrix = self._create_harmonic_matrix(in_features)
        if dynamic_harmonics:
            self.harmonic_matrix = nn.Parameter(harmonic_matrix)
        else:
            self.register_buffer("harmonic_matrix", harmonic_matrix)

        # Masking matrix for auditory masking effects
        self.register_buffer("masking_matrix", self._create_masking_matrix(in_features))

        # Learnable weight for harmonic influence
        self.harmonic_weight = nn.Parameter(torch.tensor(harmonic_weight))

    def _hz_to_mel(self, frequencies):
        """Convert Hz to Mel scale using the HTK formula."""
        return 2595 * torch.log10(1 + frequencies / 700)

    def _get_mel_frequencies(self):
        """Get center frequencies of mel bands."""
        min_mel = self._hz_to_mel(torch.tensor(20.0))
        max_mel = self._hz_to_mel(torch.tensor(self.sample_rate / 2.0))
        mels = torch.linspace(min_mel, max_mel, self.n_mels)
        return 700 * (10 ** (mels / 2595) - 1)

    def _create_harmonic_matrix(self, size):
        """Create a matrix encoding harmonic relationships based on frequency ratios."""
        matrix = torch.zeros(size, size)
        if self.is_mel_scale and self.n_mels > 0:
            frequencies = self._get_mel_frequencies()
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        ratio = frequencies[i] / frequencies[j]
                        if ratio > 1:
                            ratio = 1 / ratio
                        consonance_scores = {
                            1.0: 1.0, 0.667: 0.9, 0.75: 0.8, 0.8: 0.7,
                            0.833: 0.6, 0.889: 0.5, 0.944: 0.3
                        }
                        consonance = sum(w * torch.exp(-(ratio - r) ** 2 / (0.01 if r == 1.0 else 0.02))
                                        for r, w in consonance_scores.items())
                        matrix[i, j] = consonance
        else:
            for i in range(size):
                for j in range(size):
                    if i == j:
                        matrix[i, j] = 1.0
                    else:
                        ratio = (i + 1) / (j + 1)
                        if ratio > 1:
                            ratio = 1 / ratio
                        consonance = sum(torch.exp(-10 * (ratio - r) ** 2) 
                                        for r in [1.0, 0.667, 0.75, 0.8, 0.833])
                        matrix[i, j] = consonance
        return matrix / matrix.max()

    def _create_masking_matrix(self, size):
        """Create a matrix modeling auditory masking effects based on critical bands."""
        critical_bandwidth = max(1, int((self.n_mels if self.is_mel_scale else size) * 0.05))
        masking = torch.zeros(size, size)
        for i in range(size):
            for j in range(max(0, i - critical_bandwidth), min(size, i + critical_bandwidth + 1)):
                masking[i, j] = torch.exp(-0.5 * ((j - i) / critical_bandwidth) ** 2 if j <= i 
                                         else -1.0 * ((j - i) / critical_bandwidth) ** 2)
        return masking / masking.max()

    def forward(self, x):
        linear_out = self.linear(x)
        harmonic_influence = torch.matmul(x, torch.matmul(self.harmonic_matrix, self.linear.weight.t()))
        activations = torch.abs(x).unsqueeze(-1)
        masking_effect = (torch.matmul(activations, self.masking_matrix.unsqueeze(0)).squeeze(-1) * 0.2)
        return (linear_out + self.harmonic_weight * harmonic_influence) * (1 - masking_effect)

# --- SelfAttentionBlock ---
class SelfAttentionBlock(nn.Module):
    """Multi-head self-attention block to focus on important features."""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        orig_shape = x.shape
        if len(orig_shape) == 2:
            x = x.unsqueeze(1)
        attn_out, _ = self.attention(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.layer_norm(x)
        return x.squeeze(1) if len(orig_shape) == 2 else x

# --- TemporalBlock ---
class TemporalBlock(nn.Module):
    """Temporal modeling using GRU to capture rhythm and time-based patterns."""
    def __init__(self, input_dim, hidden_dim, num_layers=1, bidirectional=True, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, bidirectional=bidirectional,
                         dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.projection = nn.Linear(self.output_dim, input_dim)

    def forward(self, x, reshape_for_time=True):
        batch_size = x.shape[0]
        if reshape_for_time:
            time_steps = 16  # Adjustable based on data
            features_per_step = x.shape[1] // time_steps
            x = x.reshape(batch_size, time_steps, features_per_step)
        x, _ = self.gru(x)
        x = self.projection(x)
        return x.reshape(batch_size, -1) if reshape_for_time else x

# --- WaveformDecoder ---
class WaveformDecoder(nn.Module):
    """Decoder to generate raw audio waveforms from latent representations."""
    def __init__(self, latent_dim, output_samples=16384, channels=128, upsample_factors=[8, 8, 4, 2]):
        super(WaveformDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_samples = output_samples
        self.initial_projection = nn.Linear(latent_dim, channels * 4)
        self.upsample_layers = nn.ModuleList()
        current_channels = channels
        for factor in upsample_factors:
            self.upsample_layers.append(nn.Sequential(
                nn.ConvTranspose1d(current_channels, current_channels // 2, kernel_size=factor * 2,
                                  stride=factor, padding=factor // 2),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(current_channels // 2)
            ))
            current_channels //= 2
        self.final_conv = nn.Conv1d(current_channels, 1, kernel_size=7, padding=3)
        self.tanh = nn.Tanh()

    def forward(self, z):
        batch_size = z.shape[0]
        x = self.initial_projection(z).view(batch_size, -1, 4)
        for layer in self.upsample_layers:
            x = layer(x)
        waveform = self.tanh(self.final_conv(x))
        if waveform.shape[-1] != self.output_samples:
            waveform = F.interpolate(waveform, size=self.output_samples, mode='linear', align_corners=False)
        return waveform

# --- HarmonicVAE2 ---
class HarmonicVAE2(nn.Module):
    """
    Enhanced VAE for music generation with harmonic awareness, attention, and waveform generation.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256], condition_dim=0,
                 sample_rate=22050, n_mels=128, use_attention=True, use_temporal=True,
                 use_waveform=True, dynamic_harmonics=True, output_samples=16384, dropout=0.1):
        super(HarmonicVAE2, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_mel_scale = n_mels > 0
        self.condition_dim = condition_dim
        self.use_attention = use_attention
        self.use_temporal = use_temporal
        self.use_waveform = use_waveform
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.dropout = dropout

        # Conditional input processing
        if condition_dim > 0:
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, condition_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(condition_dim * 2, hidden_dims[0]),
                nn.Dropout(dropout)
            )

        # Encoder Temporal Modeling
        if use_temporal:
            self.temporal_encoder = TemporalBlock(input_dim, hidden_dims[0] // 2, bidirectional=True)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        encoder_input_dim = input_dim if not use_temporal else self.temporal_encoder.output_dim
        self.encoder_layers.append(nn.Linear(encoder_input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(HarmonicLayer(hidden_dims[i], hidden_dims[i + 1],
                                                    sample_rate=sample_rate, n_mels=n_mels,
                                                    is_mel_scale=self.is_mel_scale,
                                                    dynamic_harmonics=dynamic_harmonics))

        if use_attention:
            self.encoder_attention = SelfAttentionBlock(hidden_dims[-1], num_heads=8, dropout=dropout)

        self.latent_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.latent_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList()
        decoder_input_dim = latent_dim + (hidden_dims[0] if condition_dim > 0 else 0)
        self.decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dims[-1]))
        if use_attention:
            self.decoder_attention = SelfAttentionBlock(hidden_dims[-1], num_heads=8, dropout=dropout)
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.decoder_layers.append(HarmonicLayer(hidden_dims[i], hidden_dims[i - 1],
                                                    sample_rate=sample_rate, n_mels=n_mels,
                                                    is_mel_scale=self.is_mel_scale,
                                                    dynamic_harmonics=dynamic_harmonics))
        if use_temporal:
            self.temporal_decoder = TemporalBlock(hidden_dims[0], hidden_dims[0] // 2, bidirectional=True)
        self.output_decoder = nn.Linear(hidden_dims[0], input_dim)

        # Waveform Decoder
        if use_waveform:
            self.waveform_decoder = WaveformDecoder(decoder_input_dim, output_samples)

        self.encoder_activation = nn.LeakyReLU(0.2)
        self.decoder_activation = nn.LeakyReLU(0.2)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(dim) for dim in hidden_dims])

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, condition=None):
        if self.use_temporal:
            x = self.temporal_encoder(x)
        if condition is not None and self.condition_dim > 0:
            condition_embedding = self.condition_encoder(condition)
            x = torch.cat([x, condition_embedding], dim=1)
        for i, layer in enumerate(self.encoder_layers):
            x = self.encoder_activation(layer(x))
            if i < len(self.layer_norms):
                x = self.layer_norms[i](x)
        if self.use_attention:
            x = self.encoder_attention(x)
        mu, log_var = self.latent_mean(x), self.latent_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var

    def decode(self, z, condition=None, output_waveform=False):
        if condition is not None and self.condition_dim > 0:
            z = torch.cat([z, self.condition_encoder(condition)], dim=1)
        original_z = z
        for i, layer in enumerate(self.decoder_layers):
            z = self.decoder_activation(layer(z))
            if i == 0 and self.use_attention:
                z = self.decoder_attention(z)
        if self.use_temporal:
            z = self.temporal_decoder(z)
        spectrogram = torch.sigmoid(self.output_decoder(z))
        if output_waveform and self.use_waveform:
            waveform = self.waveform_decoder(original_z)
            return spectrogram, waveform
        return spectrogram

    def forward(self, x, condition=None):
        z, mu, log_var = self.encode(x, condition)
        if self.use_waveform:
            reconstruction, waveform = self.decode(z, condition, output_waveform=True)
            return reconstruction, waveform, z, mu, log_var
        reconstruction = self.decode(z, condition)
        return reconstruction, z, mu, log_var

    def generate(self, num_samples=1, condition=None):
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        with torch.no_grad():
            return self.decode(z, condition, output_waveform=True)[1] if self.use_waveform else self.decode(z, condition)

    def interpolate(self, x1, x2, steps=10, condition=None):
        with torch.no_grad():
            z1, _, _ = self.encode(x1.unsqueeze(0) if x1.dim() == 1 else x1, condition)
            z2, _, _ = self.encode(x2.unsqueeze(0) if x2.dim() == 1 else x2, condition)
            alphas = torch.linspace(0, 1, steps, device=z1.device)
            interpolations, waveforms = [], []
            for alpha in alphas:
                z_interp = z1 * (1 - alpha) + z2 * alpha
                if self.use_waveform:
                    spec, wave = self.decode(z_interp, condition, output_waveform=True)
                    interpolations.append(spec)
                    waveforms.append(wave)
                else:
                    interpolations.append(self.decode(z_interp, condition))
            specs = torch.cat(interpolations, dim=0)
            return (specs, torch.cat(waveforms, dim=0)) if self.use_waveform else specs

# --- PerceptualVAELoss ---
class PerceptualVAELoss(nn.Module):
    """Loss function incorporating reconstruction, spectral, KL divergence, sparsity, and perceptual losses."""
    def __init__(self, sparsity_target=0.05, sparsity_weight=0.1, kl_weight=0.01,
                 spectral_weight=0.5, perceptual_weight=1.0, waveform_weight=0.0):
        super(PerceptualVAELoss, self).__init__()
        self.sparsity_target, self.sparsity_weight = sparsity_target, sparsity_weight
        self.kl_weight, self.spectral_weight = kl_weight, spectral_weight
        self.perceptual_weight, self.waveform_weight = perceptual_weight, waveform_weight

        try:
            import torch.hub
            self.feature_extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
            self.feature_extractor.eval()
            self.has_feature_extractor = True
        except:
            print("Warning: VGGish unavailable. Perceptual loss disabled.")
            self.has_feature_extractor = False

    def spectral_loss(self, x_orig, x_recon):
        spectral_emphasis = torch.sqrt(torch.abs(x_orig) + 1e-8)
        return torch.mean(spectral_emphasis * (x_orig - x_recon) ** 2)

    def perceptual_loss(self, waveform_orig, waveform_recon):
        if not self.has_feature_extractor:
            return torch.tensor(0.0, device=waveform_orig.device)
        try:
            with torch.no_grad():
                orig_features = self.feature_extractor(waveform_orig.cpu())
                recon_features = self.feature_extractor(waveform_recon.cpu())
                return F.mse_loss(orig_features, recon_features)
        except:
            return torch.tensor(0.0, device=waveform_orig.device)

    def waveform_loss(self, waveform_orig, waveform_recon):
        return F.mse_loss(waveform_orig, waveform_recon)

    def forward(self, x_orig, x_recon, z, mu, log_var, waveform_orig=None, waveform_recon=None):
        recon_loss = F.mse_loss(x_recon, x_orig, reduction='mean')
        spec_loss = self.spectral_loss(x_orig, x_recon)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / x_orig.shape[0]
        avg_activation = torch.mean(torch.abs(z), dim=0)
        sparsity_loss = torch.sum(self.sparsity_target * torch.log((self.sparsity_target + 1e-8) / (avg_activation + 1e-8)) +
                                 (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target + 1e-8) / (1 - avg_activation + 1e-8)))
        wave_loss = self.waveform_loss(waveform_orig, waveform_recon) if self.waveform_weight > 0 and waveform_orig is not None and waveform_recon is not None else torch.tensor(0.0, device=x_orig.device)
        percep_loss = self.perceptual_loss(waveform_orig, waveform_recon) if self.perceptual_weight > 0 and waveform_orig is not None and waveform_recon is not None else torch.tensor(0.0, device=x_orig.device)
        
        total_loss = (recon_loss + self.spectral_weight * spec_loss + self.kl_weight * kl_loss +
                      self.sparsity_weight * sparsity_loss + self.waveform_weight * wave_loss +
                      self.perceptual_weight * percep_loss)
        
        return total_loss, {'total': total_loss.item(), 'recon': recon_loss.item(), 'spectral': spec_loss.item(),
                            'kl': kl_loss.item(), 'sparsity': sparsity_loss.item(), 'waveform': wave_loss.item(),
                            'perceptual': percep_loss.item()}

# --- Training Function ---
def train_harmonic_vae2(model, train_loader, test_loader=None, num_epochs=100, learning_rate=0.001,
                        beta_start=0.0001, beta_end=0.01, perceptual_weight=0.5, waveform_weight=0.5,
                        device="cuda" if torch.cuda.is_available() else "cpu", save_dir="models",
                        visualize_every=10, use_waveform=True, mixed_precision=True):
    """Train the HarmonicVAE2 model with advanced features."""
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    criterion = PerceptualVAELoss(sparsity_target=0.05, sparsity_weight=0.1, kl_weight=beta_start,
                                 spectral_weight=0.5, perceptual_weight=perceptual_weight,
                                 waveform_weight=waveform_weight).to(device)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device != 'cpu' else None

    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'beta': [], 'epoch_times': []}
    if test_loader:
        fixed_data, fixed_condition = next(iter(test_loader))[0][:8].to(device), (next(iter(test_loader))[1][:8].to(device) if len(next(iter(test_loader))) > 1 else None)

    print(f"Starting training on {device} for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        beta = beta_start + (beta_end - beta_start) * min(1.0, epoch / (0.3 * num_epochs))
        criterion.kl_weight = beta

        model.train()
        train_loss, batch_count = 0, 0
        for batch in train_loader:
            x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            condition = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
            waveform_orig = batch[2].to(device) if use_waveform and isinstance(batch, (list, tuple)) and len(batch) > 2 else None

            if mixed_precision and scaler:
                with torch.autocast(device_type='cuda'):
                    outputs = model(x, condition)
                    loss, _ = criterion(x, *outputs[:2], outputs[2], outputs[3], outputs[4], waveform_orig, outputs[1] if use_waveform else None)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(x, condition)
                loss, _ = criterion(x, *outputs[:2], outputs[2], outputs[3], outputs[4], waveform_orig, outputs[1] if use_waveform else None)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad()
            train_loss += loss.item()
            batch_count += 1

        avg_train_loss = train_loss / batch_count
        history['train_loss'].append(avg_train_loss)

        if test_loader:
            model.eval()
            val_loss, val_batch_count = 0, 0
            with torch.no_grad():
                for batch in test_loader:
                    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
                    condition = batch[1].to(device) if isinstance(batch, (list, tuple)) and len(batch) > 1 else None
                    waveform_orig = batch[2].to(device) if use_waveform and isinstance(batch, (list, tuple)) and len(batch) > 2 else None
                    outputs = model(x, condition)
                    loss, _ = criterion(x, *outputs[:2], outputs[2], outputs[3], outputs[4], waveform_orig, outputs[1] if use_waveform else None)
                    val_loss += loss.item()
                    val_batch_count += 1
            avg_val_loss = val_loss / val_batch_count
            history['val_loss'].append(avg_val_loss)
            scheduler.step(avg_val_loss)

        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['beta'].append(beta)
        history['epoch_times'].append(time.time() - epoch_start)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}" + 
              (f", Val Loss: {avg_val_loss:.4f}" if test_loader else "") +
              f", Time: {history['epoch_times'][-1]:.2f}s, LR: {history['lr'][-1]:.6f}, Beta: {beta:.6f}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"harmonic_vae2_epoch_{epoch+1}.pt")
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': avg_train_loss if test_loader is None else avg_val_loss,
                        'history': history}, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        if (epoch + 1) % visualize_every == 0 and test_loader:
            with torch.no_grad():
                outputs = model(fixed_data, fixed_condition)
                spec_recon = outputs[0]
                waveform_recon = outputs[1] if use_waveform else None
                n_mels = self.n_mels if self.is_mel_scale else 128
                time_frames = fixed_data.shape[1] // n_mels
                viz_dir = os.path.join(save_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)

                plt.figure(figsize=(15, 8))
                for i in range(min(4, fixed_data.shape[0])):
                    plt.subplot(4, 2, i * 2 + 1)
                    plt.imshow(fixed_data[i].reshape(n_mels, time_frames).cpu().numpy(), 
                              aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Original {i+1}")
                    plt.colorbar(format='%+2.0f')
                    plt.subplot(4, 2, i * 2 + 2)
                    plt.imshow(spec_recon[i].reshape(n_mels, time_frames).cpu().numpy(), 
                              aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Reconstruction {i+1}")
                    plt.colorbar(format='%+2.0f')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"reconstructions_epoch_{epoch+1}.png"))
                plt.close()

                samples = model.generate(num_samples=4, condition=fixed_condition[:4] if fixed_condition is not None else None)
                plt.figure(figsize=(15, 8))
                for i in range(min(4, samples.shape[0])):
                    plt.subplot(2, 2, i + 1)
                    if use_waveform:
                        plt.plot(samples[i].squeeze().cpu().numpy())
                        plt.title(f"Generated Waveform {i+1}")
                    else:
                        plt.imshow(samples[i].reshape(n_mels, time_frames).cpu().numpy(), 
                                  aspect='auto', origin='lower', cmap='viridis')
                        plt.title(f"Generated Sample {i+1}")
                        plt.colorbar(format='%+2.0f')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, f"generated_epoch_{epoch+1}.png"))
                plt.close()

                if use_waveform:
                    audio_dir = os.path.join(viz_dir, "audio")
                    os.makedirs(audio_dir, exist_ok=True)
                    for i, waveform in enumerate([waveform_orig[:2], waveform_recon[:2], samples[:2]]):
                        for j, w in enumerate(waveform):
                            w_np = w.squeeze().cpu().numpy()
                            w_np /= np.max(np.abs(w_np)) + 1e-8
                            wavfile.write(os.path.join(audio_dir, f"{'original' if i == 0 else 'reconstructed' if i == 1 else 'generated'}_{epoch+1}_{j+1}.wav"), 
                                         model.sample_rate, (w_np * 32767).astype(np.int16))

    final_path = os.path.join(save_dir, "harmonic_vae2_final.pt")
    torch.save({'epoch': num_epochs, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss if test_loader is None else avg_val_loss,
                'history': history}, final_path)
    print(f"Final model saved to {final_path}")
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    return model, history

# --- Example Usage ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset
    input_dim, latent_dim, condition_dim = 1024, 128, 10
    sample_rate, n_mels = 22050, 128
    batch_size = 4
    x = torch.rand(batch_size, input_dim)
    condition = torch.rand(batch_size, condition_dim)
    waveform = torch.rand(batch_size, 1, 16384)
    dataset = TensorDataset(x, condition, waveform)
    train_loader = DataLoader(dataset, batch_size=batch_size)

    model = HarmonicVAE2(input_dim, latent_dim, [512, 256], condition_dim, sample_rate, n_mels,
                        use_attention=True, use_temporal=True, use_waveform=True, dynamic_harmonics=True)
    outputs = model(x, condition)
    print(f"Spectrogram shape: {outputs[0].shape}, Waveform shape: {outputs[1].shape if model.use_waveform else 'N/A'}")

    criterion = PerceptualVAELoss()
    loss, components = criterion(x, *outputs[:2], outputs[2], outputs[3], outputs[4], waveform, outputs[1])
    print(f"Loss: {loss.item()}, Components: {components}")

    generated = model.generate(2, condition[:2])
    print(f"Generated shape: {generated.shape}")

    model, history = train_harmonic_vae2(model, train_loader, num_epochs=5, visualize_every=2)
