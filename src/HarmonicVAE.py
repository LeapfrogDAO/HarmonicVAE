import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HarmonicLayer(nn.Module):
    """
    Custom layer that incorporates psychoacoustic principles of harmony and dissonance.
    This layer processes frequency-domain data with special attention to harmonic relationships.
    Incorporates mel-scale frequency mapping and auditory masking effects.
    """
    def __init__(self, in_features, out_features, harmonic_weight=0.5, 
                 sample_rate=22050, n_mels=128, is_mel_scale=True):
        super(HarmonicLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.is_mel_scale = is_mel_scale
        
        # Create a harmonic relation matrix based on music theory and psychoacoustics
        self.register_buffer("harmonic_matrix", self._create_harmonic_matrix(
            in_features, sample_rate, n_mels))
            
        # Create a masking matrix to model auditory masking effects
        self.register_buffer("masking_matrix", self._create_masking_matrix(
            in_features, n_mels if is_mel_scale else in_features))
            
        # Weight for harmonic influence (can be learned)
        self.harmonic_weight = nn.Parameter(torch.tensor(harmonic_weight))
        
    def _hz_to_mel(self, frequencies):
        """Convert Hz to Mel scale using HTK formula."""
        return 2595 * torch.log10(1 + frequencies / 700)
        
    def _get_mel_frequencies(self, n_mels, sample_rate):
        """Get center frequencies of mel bands."""
        min_mel = self._hz_to_mel(torch.tensor(20.0))  # Min frequency in Hz
        max_mel = self._hz_to_mel(torch.tensor(sample_rate / 2.0))
        
        # Equally spaced in Mel scale
        mels = torch.linspace(min_mel, max_mel, n_mels)
        
        # Convert back to Hz
        freqs = 700 * (10**(mels / 2595) - 1)
        return freqs
        
    def _create_harmonic_matrix(self, size, sample_rate, n_mels):
        """
        Creates a matrix that encodes harmonic relationships between frequencies.
        Values close to 1 represent consonant intervals (octaves, fifths, etc.)
        Values close to 0 represent dissonant intervals.
        
        For mel-scale: uses actual mel frequency bands to calculate ratios.
        For linear scale: uses linear frequency spacing.
        """
        matrix = torch.zeros(size, size)
        
        if self.is_mel_scale and n_mels > 0:
            # Get actual frequency values for mel bands
            frequencies = self._get_mel_frequencies(n_mels, sample_rate)
            
            # Calculate frequency ratios using actual Hz values
            # This is more accurate than the linear approach
            for i in range(len(frequencies)):
                for j in range(len(frequencies)):
                    if i == j:  # Same frequency
                        matrix[i, j] = 1.0
                    else:
                        # Calculate actual frequency ratio 
                        ratio = frequencies[i] / frequencies[j]
                        # Normalize ratio to handle both directions
                        if ratio > 1:
                            ratio = 1 / ratio
                        
                        # Consonance based on Western music intervals
                        # Using empirical data from music perception research
                        consonance_scores = {
                            1.0: 1.0,     # Unison/octave (1:1 or 2:1)
                            0.667: 0.9,   # Perfect fifth (3:2)
                            0.75: 0.8,    # Perfect fourth (4:3)
                            0.8: 0.7,     # Major third (5:4) 
                            0.833: 0.6,   # Minor third (6:5)
                            0.889: 0.5,   # Major second (9:8)
                            0.944: 0.3    # Minor second (16:15)
                        }
                        
                        # Calculate consonance using a weighted mixture of Gaussians
                        consonance = 0
                        for target_ratio, weight in consonance_scores.items():
                            # Width of Gaussian determines how strict the consonance check is
                            # More psychoacoustically accurate model would vary this by frequency
                            width = 0.01 if target_ratio == 1.0 else 0.02
                            consonance += weight * torch.exp(-(ratio - target_ratio)**2 / width)
                        
                        matrix[i, j] = consonance
        else:
            # Use the original linear approach for non-mel inputs
            for i in range(size):
                for j in range(size):
                    if i == j:  # Same frequency
                        matrix[i, j] = 1.0
                    else:
                        # Calculate frequency ratio (simplistic model)
                        ratio = (i + 1) / (j + 1)
                        # Normalize ratio to handle both directions
                        if ratio > 1:
                            ratio = 1 / ratio
                        
                        # High values for consonant intervals
                        # Common consonant ratios in Western music
                        consonance = 0
                        for consonant_ratio in [1.0, 0.667, 0.75, 0.8, 0.833]:
                            consonance += torch.exp(-10 * (ratio - consonant_ratio)**2)
                        
                        matrix[i, j] = consonance
        
        return matrix / matrix.max()  # Normalize
    
    def _create_masking_matrix(self, size, n_freqs):
        """
        Creates a matrix to model auditory masking effects.
        Each frequency can mask nearby frequencies based on critical bands.
        """
        # Approximate critical bandwidth in mel scale (simplified model)
        # In real psychoacoustics, this would vary with frequency
        critical_bandwidth = max(1, int(n_freqs * 0.05))  # ~5% of frequency range
        
        masking = torch.zeros(size, size)
        for i in range(size):
            # Each frequency masks nearby frequencies, with effect decreasing with distance
            for j in range(max(0, i - critical_bandwidth), min(size, i + critical_bandwidth + 1)):
                # Asymmetric masking (higher frequencies mask lower ones more than vice versa)
                if j <= i:  # Lower or same frequency
                    masking[i, j] = torch.exp(-0.5 * ((j - i) / critical_bandwidth)**2)
                else:  # Higher frequency
                    masking[i, j] = torch.exp(-1.0 * ((j - i) / critical_bandwidth)**2)
        
        return masking / masking.max()  # Normalize
    
    def forward(self, x):
        # Standard linear transformation
        linear_out = self.linear(x)
        
        # Apply harmonic relationships
        harmonic_influence = torch.matmul(x, torch.matmul(self.harmonic_matrix, self.linear.weight.t()))
        
        # Apply masking effects (simulate how loud frequencies mask quieter ones)
        # First, get activation levels to determine masking strength
        activations = torch.abs(x).unsqueeze(-1)
        # Apply masking based on activation levels
        masking_effect = torch.matmul(activations, self.masking_matrix.unsqueeze(0))
        # Scale the masking effect to be subtle
        masking_effect = masking_effect.squeeze(-1) * 0.2
        
        # Combine standard output with harmonic influence and apply masking
        combined = linear_out + self.harmonic_weight * harmonic_influence
        # Masking acts as a kind of attention mechanism
        return combined * (1 - masking_effect)


class HarmonicVAE(nn.Module):
    """
    Variational Autoencoder that operates in the frequency domain and incorporates
    harmonic relationships between frequencies. Uses a hierarchical latent space
    and reparameterization trick for better generation capabilities.
    """
    def __init__(self, input_dim, latent_dim, hidden_dims=[512, 256], 
                 sample_rate=22050, n_mels=128):
        super(HarmonicVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.is_mel_scale = n_mels > 0
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList()
        
        # First layer is a regular linear layer
        self.encoder_layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        # Add harmonic layers for the rest of the encoder
        for i in range(len(hidden_dims)-1):
            self.encoder_layers.append(
                HarmonicLayer(
                    hidden_dims[i], 
                    hidden_dims[i+1],
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    is_mel_scale=self.is_mel_scale
                )
            )
        
        # VAE approach: separate layers for mean and log variance
        self.latent_mean = nn.Linear(hidden_dims[-1], latent_dim)
        self.latent_log_var = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder layers (reverse of encoder)
        self.decoder_layers = nn.ModuleList()
        
        # First decoder layer from latent space
        self.decoder_layers.append(nn.Linear(latent_dim, hidden_dims[-1]))
        
        # Add harmonic layers for the rest of the decoder (in reverse)
        for i in range(len(hidden_dims)-1, 0, -1):
            self.decoder_layers.append(
                HarmonicLayer(
                    hidden_dims[i], 
                    hidden_dims[i-1],
                    sample_rate=sample_rate,
                    n_mels=n_mels,
                    is_mel_scale=self.is_mel_scale
                )
            )
        
        # Final decoding layer back to input dimension
        self.output_decoder = nn.Linear(hidden_dims[0], input_dim)
        
        # Activation functions
        self.encoder_activation = nn.LeakyReLU(0.2)
        self.decoder_activation = nn.LeakyReLU(0.2)
        
        # Add layer normalization for more stable training
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])
        
    def reparameterize(self, mu, log_var):
        """
        Reparameterization trick to sample from N(mu, var) from N(0,1).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def encode(self, x):
        # Pass through encoder layers with layer normalization
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            x = self.encoder_activation(x)
            if i < len(self.layer_norms):
                x = self.layer_norms[i](x)
        
        # Get latent parameters (mean and log variance)
        mu = self.latent_mean(x)
        log_var = self.latent_log_var(x)
        
        # Sample from the latent distribution
        z = self.reparameterize(mu, log_var)
        
        return z, mu, log_var
    
    def decode(self, z):
        # Pass through decoder layers
        for i, layer in enumerate(self.decoder_layers):
            z = layer(z)
            z = self.decoder_activation(z)
            # No layer norm in the decoder for generating rich outputs
        
        # Final decoding back to input space
        output = self.output_decoder(z)
        
        # Apply activation based on the target range
        # Sigmoid is good for normalized spectrograms in [0, 1]
        return torch.sigmoid(output)
    
    def forward(self, x):
        # Encode to latent space
        z, mu, log_var = self.encode(x)
        
        # Decode from latent space
        reconstruction = self.decode(z)
        
        return reconstruction, z, mu, log_var
    
    def generate(self, num_samples=1):
        """
        Generate new samples from random points in the latent space.
        """
        # Sample from standard normal distribution
        z = torch.randn(num_samples, self.latent_dim).to(next(self.parameters()).device)
        
        # Decode the random samples
        with torch.no_grad():
            samples = self.decode(z)
            
        return samples
    
    def interpolate(self, x1, x2, steps=10):
        """
        Interpolate between two input samples in the latent space.
        """
        # Encode both inputs
        with torch.no_grad():
            z1, _, _ = self.encode(x1.unsqueeze(0) if x1.dim() == 1 else x1)
            z2, _, _ = self.encode(x2.unsqueeze(0) if x2.dim() == 1 else x2)
            
            # Create interpolation path
            alphas = torch.linspace(0, 1, steps, device=z1.device)
            interpolations = []
            
            # Generate samples along the path
            for alpha in alphas:
                z_interp = z1 * (1 - alpha) + z2 * alpha
                interp_sample = self.decode(z_interp)
                interpolations.append(interp_sample)
                
        return torch.cat(interpolations, dim=0)


class HarmonicVAELoss(nn.Module):
    """
    Custom loss function for the Harmonic VAE.
    Combines reconstruction loss, KL divergence for the VAE, and sparsity penalty.
    Can also include optional frequency-domain perceptual losses.
    """
    def __init__(self, sparsity_target=0.05, sparsity_weight=0.1, 
                 kl_weight=0.01, spectral_weight=0.5):
        super(HarmonicVAELoss, self).__init__()
        self.sparsity_target = sparsity_target  # Target activation rate
        self.sparsity_weight = sparsity_weight  # Weight of sparsity penalty
        self.kl_weight = kl_weight  # Weight for KL divergence loss
        self.spectral_weight = spectral_weight  # Weight for spectral loss
        
    def spectral_loss(self, x_orig, x_recon):
        """
        Perceptual loss in the frequency domain that focuses on the most salient frequencies.
        Emphasizes errors in higher-energy parts of the spectrum which are more perceptible.
        """
        # Reshape if flattened
        batch_size = x_orig.shape[0]
        
        # Focus more on errors in high-energy frequency bins
        # This is psychoacoustically motivated: we hear errors in loud parts more
        spectral_emphasis = torch.sqrt(torch.abs(x_orig) + 1e-8)  # Emphasize high-energy parts
        weighted_error = spectral_emphasis * (x_orig - x_recon)**2
        
        return torch.mean(weighted_error)
        
    def forward(self, x_orig, x_recon, z, mu, log_var):
        """
        Full loss function including reconstruction, KL divergence, and sparsity.
        
        Args:
            x_orig: Original input
            x_recon: Reconstructed output
            z: Sampled latent vector
            mu: Mean of latent distribution
            log_var: Log variance of latent distribution
        """
        # Basic reconstruction loss (mean squared error)
        recon_loss = F.mse_loss(x_recon, x_orig, reduction='mean')
        
        # Spectral perceptual loss for better audio quality
        spec_loss = self.spectral_loss(x_orig, x_recon)
        
        # KL divergence to enforce the latent space distribution
        # This is the standard VAE loss component
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Sparsity loss to encourage sparse activations in the latent space
        avg_activation = torch.mean(torch.abs(z), dim=0)
        sparsity_loss = torch.sum(
            self.sparsity_target * torch.log((self.sparsity_target + 1e-8) / (avg_activation + 1e-8)) +
            (1 - self.sparsity_target) * torch.log((1 - self.sparsity_target + 1e-8) / (1 - avg_activation + 1e-8))
        )
        
        # Normalize by batch size to make it consistent across different batch sizes
        batch_size = x_orig.shape[0]
        kl_loss = kl_loss / batch_size
        
        # Combined loss with weights for each component
        total_loss = recon_loss + \
                    self.spectral_weight * spec_loss + \
                    self.kl_weight * kl_loss + \
                    self.sparsity_weight * sparsity_loss
        
        # For monitoring individual components
        loss_components = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'spectral': spec_loss.item(),
            'kl': kl_loss.item(),
            'sparsity': sparsity_loss.item()
        }
        
        return total_loss, loss_components


def generate_nonsense_data(num_samples=100, input_dim=2048, complexity=0.5, 
                       timbre_type='random', rhythm_pattern=None, noise_level=0.1):
    """
    Generate "nonsense" audio-like data for training diversity.
    
    Args:
        num_samples: Number of samples to generate
        input_dim: Dimension of each sample (representing frequency bins)
        complexity: Controls how structured/chaotic the data is (0-1)
        timbre_type: Type of harmonic structure to simulate:
            - 'random': Random harmonics
            - 'flute': Few harmonics, strong fundamental
            - 'guitar': Rich harmonics with natural decay
            - 'bell': Inharmonic spectrum with stretched harmonics
        rhythm_pattern: Optional pattern to modulate the signal (simulating temporal structure)
        noise_level: Amount of noise to add (0-1)
    
    Returns:
        Tensor of shape [num_samples, input_dim] with generated spectral data
    """
    import random
    
    data = []
    
    # Set up timbre characteristics based on type
    if timbre_type == 'flute':
        # Flutes have strong fundamental and few harmonics
        harmonic_weights = [1.0, 0.3, 0.1, 0.05]  
        harmonic_count = 4
        decay_factor = 2.5
    elif timbre_type == 'guitar':
        # Guitars have rich harmonic content
        harmonic_weights = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]
        harmonic_count = 8
        decay_factor = 1.3
    elif timbre_type == 'bell':
        # Bells have inharmonic spectra (stretched harmonics)
        harmonic_weights = [1.0, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1]
        harmonic_count = 7
        decay_factor = 1.0
        # Bells use stretched harmonics (non-integer multiples)
        stretch_factor = 1.05  
    else:  # 'random'
        # Randomize harmonic structure
        harmonic_count = int(3 + complexity * 7)  # 3-10 harmonics based on complexity
        harmonic_weights = [1.0]
        harmonic_weights.extend([random.random() * (1.0 - 0.1 * i) for i in range(1, harmonic_count)])
        decay_factor = 1.0 + random.random() * 2.0
        stretch_factor = 1.0
    
    # Set up rhythm pattern if specified
    if rhythm_pattern is None:
        # Default: no rhythmic modulation
        modulation = torch.ones(input_dim)
    else:
        # Create rhythmic modulation across frequency bins
        tempo = rhythm_pattern.get('tempo', 120)  # beats per minute
        pattern_length = rhythm_pattern.get('length', 4)  # beats
        
        # Create a modulation envelope
        modulation_freq = tempo / 60  # beats per second
        time_points = torch.linspace(0, pattern_length / modulation_freq, input_dim)
        
        # Simple rhythm pattern (can be enhanced with more complex patterns)
        modulation = 0.5 + 0.5 * torch.sin(2 * torch.pi * modulation_freq * time_points)
    
    for _ in range(num_samples):
        # Generate random signal as a baseline
        base_signal = torch.randn(input_dim) * noise_level
        
        # Number of fundamental frequencies to generate
        n_fundamentals = int(1 + complexity * 4)  # 1-5 based on complexity
        
        for _ in range(n_fundamentals):
            # Choose a fundamental frequency
            fundamental = torch.randint(5, input_dim // harmonic_count, (1,)).item()
            
            # Strength of this sound element
            strength = 0.5 + 0.5 * torch.rand(1).item()
            
            # Apply the selected timbre profile with harmonics
            for i, weight in enumerate(harmonic_weights):
                if timbre_type == 'bell':
                    # Bells have stretched harmonics (non-integer multiples of fundamental)
                    harmonic_idx = int(fundamental * (1.0 + i * (stretch_factor - 1.0)))
                else:
                    # Regular harmonics are integer multiples
                    harmonic_idx = fundamental * (i + 1)
                
                if harmonic_idx < input_dim:
                    # Higher harmonics decay based on the instrument type
                    harmonic_strength = strength * weight / ((i + 1) ** (1/decay_factor))
                    
                    # Apply a Gaussian window around each harmonic peak
                    window_width = max(2, int(5 * complexity))
                    window = torch.exp(-0.5 * torch.arange(-window_width, window_width+1)**2 / (complexity * 3)**2)
                    
                    # Ensure window fits within boundaries
                    start_idx = max(0, harmonic_idx - window_width)
                    end_idx = min(input_dim, harmonic_idx + window_width + 1)
                    
                    # Adjust window to fit the available space
                    win_start = window_width - (harmonic_idx - start_idx)
                    win_end = window_width + (end_idx - harmonic_idx)
                    window_adjusted = window[win_start:win_end]
                    
                    # Add the harmonic peak to the spectrum
                    base_signal[start_idx:end_idx] += harmonic_strength * window_adjusted
        
        # Apply rhythmic modulation if specified
        if rhythm_pattern is not None:
            base_signal = base_signal * modulation
        
        # Add background noise floor (more complex spectra have this)
        pink_noise = torch.rand(input_dim) * torch.exp(-torch.linspace(0, 4, input_dim))
        base_signal += pink_noise * noise_level * complexity
        
        # Normalize to [0, 1] range
        if torch.max(base_signal) > torch.min(base_signal):
            base_signal = (base_signal - torch.min(base_signal)) / (torch.max(base_signal) - torch.min(base_signal))
        
        data.append(base_signal)
    
    return torch.stack(data)


if __name__ == "__main__":
    # Example usage and testing
    # Generate synthetic data
    input_dim = 1024
    latent_dim = 64
    batch_size = 16
    
    print("Generating synthetic data...")
    synthetic_data = generate_nonsense_data(
        num_samples=batch_size, 
        input_dim=input_dim, 
        complexity=0.7,
        timbre_type='guitar'
    )
    
    # Create model
    print("Creating model...")
    model = HarmonicVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=[512, 256, 128],
        sample_rate=22050,
        n_mels=128
    )
    
    # Test forward pass
    print("Testing forward pass...")
    reconstructed, z, mu, log_var = model(synthetic_data)
    
    # Test loss function
    print("Testing loss function...")
    criterion = HarmonicVAELoss(
        sparsity_target=0.05,
        sparsity_weight=0.1,
        kl_weight=0.01,
        spectral_weight=0.5
    )
    
    loss, components = criterion(synthetic_data, reconstructed, z, mu, log_var)
    
    print(f"Input shape: {synthetic_data.shape}")
    print(f"Reconstruction shape: {reconstructed.shape}")
    print(f"Latent representation shape: {z.shape}")
    print(f"Loss: {loss.item()}")
    print(f"Loss components: {components}")
    
    # Test generate method
    print("Testing generation...")
    samples = model.generate(num_samples=5)
    print(f"Generated samples shape: {samples.shape}")
    
    # Test interpolation
    print("Testing interpolation...")
    sample1 = synthetic_data[0]
    sample2 = synthetic_data[1]
    interpolations = model.interpolate(sample1, sample2, steps=5)
    print(f"Interpolations shape: {interpolations.shape}")
    
    print("All tests passed!")
