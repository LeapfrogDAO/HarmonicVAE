import os
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import glob
import random
import argparse
from sklearn.manifold import TSNE

# Assuming HarmonicVAE and HarmonicVAELoss are defined in a separate module
# For this example, we'll include placeholder definitions
class HarmonicVAE(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims=[1024, 512, 256], sample_rate=22050, n_mels=128):
        super(HarmonicVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(torch.nn.Linear(prev_dim, h_dim))
            encoder_layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        self.encoder = torch.nn.Sequential(*encoder_layers)
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = torch.nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(torch.nn.Linear(prev_dim, h_dim))
            decoder_layers.append(torch.nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(torch.nn.Linear(prev_dim, input_dim))
        decoder_layers.append(torch.nn.Sigmoid())  # Assuming normalized input
        self.decoder = torch.nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, z, mu, log_var

class HarmonicVAELoss(torch.nn.Module):
    def __init__(self, sparsity_target=0.05, sparsity_weight=0.1, kl_weight=0.01, spectral_weight=0.5):
        super(HarmonicVAELoss, self).__init__()
        self.sparsity_target = sparsity_target
        self.sparsity_weight = sparsity_weight
        self.kl_weight = kl_weight
        self.spectral_weight = spectral_weight
        self.recon_loss_fn = torch.nn.MSELoss(reduction='sum')

    def forward(self, x, recon_x, z, mu, log_var):
        # Reconstruction loss
        recon_loss = self.recon_loss_fn(recon_x, x)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Sparsity regularization (L1 on latent z)
        sparsity_loss = torch.mean(torch.abs(z))

        # Total loss
        total_loss = (recon_loss + self.kl_weight * kl_loss + 
                      self.sparsity_weight * sparsity_loss)

        components = {
            'recon': recon_loss.item(),
            'kl': kl_loss.item(),
            'sparsity': sparsity_loss.item()
        }
        return total_loss, components

#
class RealAudioDataset(Dataset):
    """
    Dataset for loading and preprocessing real audio files into mel spectrograms.
    Handles various audio formats and lengths, with random cropping for consistent sizes.
    """
    def __init__(self, audio_paths, segment_length=4.0, sample_rate=22050, n_mels=128, 
                 n_fft=2048, hop_length=512, normalize=True, augment=False):
        """
        Initialize the dataset with audio file paths and processing parameters.
        
        Args:
            audio_paths (list): List of paths to audio files
            segment_length (float): Length of audio segments in seconds
            sample_rate (int): Target sample rate for all audio
            n_mels (int): Number of mel bands to generate
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            normalize (bool): Whether to normalize spectrograms to [0,1]
            augment (bool): Whether to apply data augmentation
        """
        self.audio_paths = audio_paths
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.segment_samples = int(segment_length * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.augment = augment
        
        # Create mel spectrogram transform
        self.mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=sample_rate//2,
            norm='slaney',
            mel_scale='htk'
        )
        
        # Dynamic range compression
        self.dynamic_range_compression = lambda x: torch.log1p(x)
        
        # Filter out invalid files
        self._validate_files()
        
        # Store metadata
        self.metadata = self._extract_metadata()
    
    def _validate_files(self):
        """Filter audio files to keep only those that can be loaded and meet minimum length."""
        valid_files = []
        min_length = self.segment_samples
        
        print(f"Validating {len(self.audio_paths)} audio files...")
        for path in tqdm(self.audio_paths):
            try:
                metadata = torchaudio.info(path)
                if metadata.num_frames >= min_length:
                    valid_files.append(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {str(e)}")
                continue
        
        self.audio_paths = valid_files
        print(f"Kept {len(valid_files)} valid audio files.")
    
    def _extract_metadata(self):
        """Extract metadata from filenames or directories."""
        metadata = {}
        for path in self.audio_paths:
            genre = os.path.basename(os.path.dirname(path))
            metadata[path] = {'genre': genre if genre else 'unknown'}
        return metadata
    
    def __len__(self):
        return len(self.audio_paths)
    
    def _load_audio(self, path):
        """Load audio file with proper resampling."""
        try:
            waveform, sr = torchaudio.load(path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
            return waveform
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.randn(1, self.segment_samples) * 0.01
    
    def _apply_augmentation(self, waveform):
        """Apply audio augmentations to the waveform."""
        if not self.augment:
            return waveform
        
        # Random gain
        if random.random() > 0.5:
            gain = random.uniform(0.5, 1.5)
            waveform = waveform * gain
        
        # Random time stretching
        if random.random() > 0.7:
            stretch_factor = random.uniform(0.8, 1.2)
            effects = [['tempo', str(stretch_factor)], ['rate', str(self.sample_rate)]]
            try:
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, self.sample_rate, effects, channels_first=True
                )
            except:
                pass
        
        # Random pitch shifting
        if random.random() > 0.7:
            semitones = random.uniform(-2, 2)
            effects = [['pitch', str(semitones * 100)], ['rate', str(self.sample_rate)]]
            try:
                waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                    waveform, self.sample_rate, effects, channels_first=True
                )
            except:
                pass
        
        return waveform
    
    def __getitem__(self, idx):
        """Get a random segment of the audio file as a mel spectrogram."""
        audio_path = self.audio_paths[idx]
        waveform = self._load_audio(audio_path)
        
        if self.augment:
            waveform = self._apply_augmentation(waveform)
        
        if waveform.shape[1] > self.segment_samples:
            start = random.randint(0, waveform.shape[1] - self.segment_samples)
            waveform = waveform[:, start:start + self.segment_samples]
        else:
            padding = self.segment_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        mel_spec = self.mel_spec_transform(waveform)
        log_mel_spec = self.dynamic_range_compression(mel_spec)
        
        if self.normalize:
            min_val, max_val = torch.min(log_mel_spec), torch.max(log_mel_spec)
            if max_val > min_val:
                log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val)
        
        return log_mel_spec.reshape(-1)

def setup_maestro_dataset(maestro_dir, batch_size=32, **kwargs):
    """
    Setup the MAESTRO dataset with train/test splits based on metadata.
    
    Args:
        maestro_dir (str): Directory containing the MAESTRO dataset
        batch_size (int): Batch size for DataLoader
        **kwargs: Additional parameters for RealAudioDataset
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    import pandas as pd
    
    maestro_csv = os.path.join(maestro_dir, "maestro-v3.0.0.csv")
    if not os.path.exists(maestro_csv):
        raise FileNotFoundError(f"MAESTRO metadata file not found: {maestro_csv}")
    
    metadata = pd.read_csv(maestro_csv)
    audio_paths = [os.path.join(maestro_dir, row['audio_filename']) 
                   for _, row in metadata.iterrows() if os.path.exists(os.path.join(maestro_dir, row['audio_filename']))]
    
    train_paths = [path for path, year in zip(audio_paths, metadata['year']) if year not in [2006, 2011]]
    test_paths = [path for path, year in zip(audio_paths, metadata['year']) if year in [2006, 2011]]
    
    train_dataset = RealAudioDataset(train_paths, augment=True, **kwargs)
    test_dataset = RealAudioDataset(test_paths, augment=False, **kwargs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"MAESTRO dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader

def setup_gtzan_dataset(gtzan_dir, batch_size=32, test_size=0.2, **kwargs):
    """
    Setup the GTZAN dataset with random train/test split.
    
    Args:
        gtzan_dir (str): Directory containing the GTZAN dataset
        batch_size (int): Batch size for DataLoader
        test_size (float): Fraction of data for testing
        **kwargs: Additional parameters for RealAudioDataset
    
    Returns:
        tuple: (train_loader, test_loader)
    """
    audio_paths = []
    for ext in ['.wav', '.au', '.mp3']:
        audio_paths.extend(glob.glob(os.path.join(gtzan_dir, f"**/*{ext}"), recursive=True))
    
    train_paths, test_paths = train_test_split(audio_paths, test_size=test_size, random_state=42)
    
    train_dataset = RealAudioDataset(train_paths, augment=True, **kwargs)
    test_dataset = RealAudioDataset(test_paths, augment=False, **kwargs)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"GTZAN dataset loaded: {len(train_dataset)} train, {len(test_dataset)} test samples")
    return train_loader, test_loader

def visualization_from_audio(audio_path, vae_model, sample_rate=22050, segment_length=4.0, 
                            n_mels=128, n_fft=2048, hop_length=512):
    """
    Visualize the encoding and reconstruction of an audio file.
    
    Args:
        audio_path (str): Path to audio file
        vae_model (HarmonicVAE): Trained model
        sample_rate (int): Sample rate
        segment_length (float): Segment length in seconds
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop length
    
    Returns:
        np.ndarray: Latent representation
    """
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    
    segment_samples = int(segment_length * sample_rate)
    if waveform.shape[1] > segment_samples:
        start = (waveform.shape[1] - segment_samples) // 2
        waveform = waveform[:, start:start + segment_samples]
    else:
        padding = segment_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        f_min=20, f_max=sample_rate//2, norm='slaney', mel_scale='htk'
    )
    
    mel_spec = mel_transform(waveform)
    log_mel_spec = torch.log1p(mel_spec)
    min_val, max_val = torch.min(log_mel_spec), torch.max(log_mel_spec)
    if max_val > min_val:
        log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val)
    
    model_input = log_mel_spec.reshape(1, -1).to(next(vae_model.parameters()).device)
    
    with torch.no_grad():
        recon, _, mu, log_var = vae_model(model_input)
        recon_melspec = recon.view(log_mel_spec.shape)
    
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 1, 1)
    plt.title("Original Mel Spectrogram")
    plt.imshow(log_mel_spec[0].cpu(), aspect='auto', origin='lower', cmap='viridis')
    plt.ylabel("Mel Bands")
    plt.colorbar(format='%+2.0f')
    
    plt.subplot(2, 1, 2)
    plt.title("Reconstructed Mel Spectrogram")
    plt.imshow(recon_melspec[0].cpu(), aspect='auto', origin='lower', cmap='viridis')
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.colorbar(format='%+2.0f')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Latent Stats - Mean: {mu.mean():.4f}, Std: {mu.std():.4f}, Sparsity: {((mu.abs() < 0.1).float().mean()*100):.2f}%")
    return mu.cpu().numpy()

def audio_from_latent(vae_model, latent_vector, output_path, n_mels=128, segment_length=4.0, 
                      sample_rate=22050, n_fft=2048, hop_length=512):
    """
    Generate audio from a latent vector.
    
    Args:
        vae_model (HarmonicVAE): Trained model
        latent_vector (torch.Tensor): Latent vector
        output_path (str): Path to save audio
        n_mels (int): Number of mel bands
        segment_length (float): Segment length in seconds
        sample_rate (int): Sample rate
        n_fft (int): FFT window size
        hop_length (int): Hop length
    
    Returns:
        str: Path to saved audio
    """
    if not isinstance(latent_vector, torch.Tensor):
        latent_vector = torch.tensor(latent_vector, dtype=torch.float32)
    if len(latent_vector.shape) == 1:
        latent_vector = latent_vector.unsqueeze(0)
    
    device = next(vae_model.parameters()).device
    latent_vector = latent_vector.to(device)
    
    with torch.no_grad():
        generated_spec = vae_model.decode(latent_vector)
    
    time_frames = int(segment_length * sample_rate / hop_length)
    generated_spec = generated_spec.reshape(1, n_mels, time_frames)
    generated_spec = torch.exp(generated_spec) - 1  # Inverse log1p
    
    try:
        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft//2 + 1, n_mels=n_mels, sample_rate=sample_rate,
            f_min=20, f_max=sample_rate//2, norm='slaney', mel_scale='htk'
        )
        linear_spec = inverse_mel(generated_spec.to(device))
        
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, power=1.0, n_iter=32
        )
        waveform = griffin_lim(linear_spec)
    except Exception as e:
        print(f"Griffin-Lim failed: {e}. Using fallback.")
        random_phase = torch.rand_like(generated_spec) * 2 * np.pi
        complex_spec = generated_spec * torch.exp(1j * random_phase)
        waveform = torch.istft(complex_spec, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
    
    waveform = waveform / torch.max(torch.abs(waveform))
    torchaudio.save(output_path, waveform.cpu(), sample_rate)
    return output_path

def train_vae_on_real_data(vae_model, train_loader, test_loader, num_epochs=50, learning_rate=0.001, 
                           save_dir="models", visualize_every=10):
    """
    Train the Harmonic VAE on real audio data.
    
    Args:
        vae_model (HarmonicVAE): Model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        num_epochs (int): Number of epochs
        learning_rate (float): Learning rate
        save_dir (str): Directory to save models
        visualize_every (int): Visualize every N epochs
    
    Returns:
        tuple: (trained model, training history)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = vae_model.to(device)
    
    start_beta, end_beta = 0.0001, 0.01
    criterion = HarmonicVAELoss(kl_weight=start_beta)
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'beta': []}
    fixed_data = next(iter(test_loader))[:8].to(device)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        beta = start_beta + (end_beta - start_beta) * min(1.0, epoch / (0.3 * num_epochs))
        criterion.kl_weight = beta
        
        vae_model.train()
        train_loss = 0
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            recon, z, mu, log_var = vae_model(batch_data)
            loss, _ = criterion(batch_data, recon, z, mu, log_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        vae_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_data in test_loader:
                batch_data = batch_data.to(device)
                recon, z, mu, log_var = vae_model(batch_data)
                loss, _ = criterion(batch_data, recon, z, mu, log_var)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['beta'].append(beta)
        
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(vae_model.state_dict(), os.path.join(save_dir, "best_model.pt"))
        
        if (epoch + 1) % visualize_every == 0:
            with torch.no_grad():
                recon, _, _, _ = vae_model(fixed_data)
                n_mels = 128
                time_frames = fixed_data.shape[1] // n_mels
                plt.figure(figsize=(15, 10))
                for i in range(min(4, fixed_data.shape[0])):
                    plt.subplot(4, 2, i*2 + 1)
                    plt.imshow(fixed_data[i].reshape(n_mels, time_frames).cpu(), aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Original {i+1}")
                    plt.subplot(4, 2, i*2 + 2)
                    plt.imshow(recon[i].reshape(n_mels, time_frames).cpu(), aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Recon {i+1}")
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"recon_epoch_{epoch+1}.png"))
                plt.close()
    
    return vae_model, history

def visualize_latent_space(vae_model, audio_files, output_path, sample_rate=22050, segment_length=4.0, n_mels=128):
    """
    Visualize the latent space using t-SNE.
    
    Args:
        vae_model (HarmonicVAE): Trained model
        audio_files (list): List of audio file paths
        output_path (str): Path to save visualization
        sample_rate (int): Sample rate
        segment_length (float): Segment length
        n_mels (int): Number of mel bands
    """
    dataset = RealAudioDataset(audio_files, segment_length=segment_length, sample_rate=sample_rate, n_mels=n_mels, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    latents = []
    genres = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(next(vae_model.parameters()).device)
            _, _, mu, _ = vae_model(batch)
            latents.append(mu.cpu().numpy())
            genres.extend([dataset.metadata[dataset.audio_paths[i]]['genre'] for i in range(len(dataset.audio_paths))][:len(mu)])
    
    latents = np.concatenate(latents, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    latents_2d = tsne.fit_transform(latents)
    
    unique_genres = sorted(set(genres))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_genres)))
    plt.figure(figsize=(12, 10))
    for genre, color in zip(unique_genres, colors):
        idx = [i for i, g in enumerate(genres) if g == genre]
        plt.scatter(latents_2d[idx, 0], latents_2d[idx, 1], c=[color], label=genre, alpha=0.6)
    plt.legend()
    plt.title("Latent Space Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(output_path)
    plt.close()

def interpolate_between_audio(vae_model, audio_path1, audio_path2, output_dir, steps=5, **kwargs):
    """
    Interpolate between two audio files in the latent space.
    
    Args:
        vae_model (HarmonicVAE): Trained model
        audio_path1 (str): First audio file
        audio_path2 (str): Second audio file
        output_dir (str): Directory to save interpolations
        steps (int): Number of interpolation steps
        **kwargs: Parameters for visualization_from_audio and audio_from_latent
    """
    os.makedirs(output_dir, exist_ok=True)
    
    latent1 = visualization_from_audio(audio_path1, vae_model, **kwargs)
    latent2 = visualization_from_audio(audio_path2, vae_model, **kwargs)
    
    for i in range(steps + 1):
        alpha = i / steps
        interp_latent = (1 - alpha) * latent1 + alpha * latent2
        output_path = os.path.join(output_dir, f"interp_step_{i}.wav")
        audio_from_latent(vae_model, interp_latent, output_path, **kwargs)
        print(f"Saved interpolation step {i} to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Harmonic VAE for Music Audio')
    parser.add_argument('--dataset', choices=['maestro', 'gtzan', 'custom'], default='gtzan', help='Dataset to use')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--custom_dir', type=str, default=None, help='Directory with custom audio files')
    parser.add_argument('--mode', choices=['train', 'generate', 'visualize', 'interpolate'], default='train', 
                        help='Operation mode')
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--latent_dim', type=int, default=128, help='Latent space dimension')
    parser.add_argument('--segment_length', type=float, default=4.0, help='Audio segment length in seconds')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--audio1', type=str, default=None, help='First audio file for interpolation')
    parser.add_argument('--audio2', type=str, default=None, help='Second audio file for interpolation')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'train':
        if args.dataset == 'maestro':
            train_loader, test_loader = setup_maestro_dataset(args.dataset_dir, batch_size=args.batch_size, 
                                                              segment_length=args.segment_length)
        elif args.dataset == 'gtzan':
            train_loader, test_loader = setup_gtzan_dataset(args.dataset_dir, batch_size=args.batch_size, 
                                                            segment_length=args.segment_length)
        else:
            if not args.custom_dir:
                raise ValueError("Custom dataset requires --custom_dir")
            audio_files = glob.glob(os.path.join(args.custom_dir, '**/*.wav'), recursive=True)
            train_files, test_files = train_test_split(audio_files, test_size=0.2, random_state=42)
            train_dataset = RealAudioDataset(train_files, segment_length=args.segment_length)
            test_dataset = RealAudioDataset(test_files, segment_length=args.segment_length)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        input_dim = next(iter(train_loader)).shape[1]
        model = HarmonicVAE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
        model, history = train_vae_on_real_data(model, train_loader, test_loader, num_epochs=args.epochs, 
                                                save_dir=args.output_dir)
    
    elif args.mode in ['generate', 'visualize', 'interpolate']:
        if not args.model_path:
            raise ValueError(f"{args.mode} mode requires --model_path")
        
        checkpoint = torch.load(args.model_path, map_location=device)
        input_dim = next(iter(RealAudioDataset([args.dataset_dir + '/**/*.wav'], 
                                               segment_length=args.segment_length)).__getitem__(0)).shape[0]
        model = HarmonicVAE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
        model.load_state_dict(checkpoint)
        model.eval()
        
        if args.mode == 'generate':
            os.makedirs(os.path.join(args.output_dir, 'generated'), exist_ok=True)
            for i in range(10):
                z = torch.randn(1, args.latent_dim).to(device)
                audio_from_latent(model, z, os.path.join(args.output_dir, 'generated', f'sample_{i}.wav'), 
                                  segment_length=args.segment_length)
        
        elif args.mode == 'visualize':
            audio_files = glob.glob(os.path.join(args.dataset_dir, '**/*.wav'), recursive=True)
            visualize_latent_space(model, audio_files, os.path.join(args.output_dir, 'latent_space.png'), 
                                   segment_length=args.segment_length)
        
        elif args.mode == 'interpolate':
            if not (args.audio1 and args.audio2):
                raise ValueError("Interpolation requires --audio1 and --audio2")
            interpolate_between_audio(model, args.audio1, args.audio2, os.path.join(args.output_dir, 'interpolations'), 
                                      segment_length=args.segment_length)

if __name__ == "__main__":
    main()
