import os
import torch
import numpy as np
import gradio as gr
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, InverseMelScale, GriffinLim

# Import the model
from src.HarmonicVAE import HarmonicVAE
from harmonic_vae_upgrades import HarmonicVAE2

class HarmonicVAEInterface:
    def __init__(self, model_path, model_version=2, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_version = model_version
        
        # Load model checkpoint
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            raise ValueError(f"Failed to load model checkpoint: {e}")
        
        # Extract model parameters
        if 'model_hyperparams' in checkpoint:
            hyperparams = checkpoint['model_hyperparams']
            self.input_dim = hyperparams.get('input_dim', 1024)
            self.latent_dim = hyperparams.get('latent_dim', 128)
            self.sample_rate = hyperparams.get('sample_rate', 22050)
            self.n_mels = hyperparams.get('n_mels', 128)
            self.hidden_dims = hyperparams.get('hidden_dims', [512, 256])
            self.n_fft = hyperparams.get('n_fft', 2048)
            self.hop_length = hyperparams.get('hop_length', 512)
        else:
            self.input_dim = 1024
            self.latent_dim = 128
            self.sample_rate = 22050
            self.n_mels = 128
            self.n_fft = 2048
            self.hop_length = 512
            self.hidden_dims = [512, 256]
        
        # Create model instance
        if model_version == 2:
            self.model = HarmonicVAE2(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                condition_dim=10,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels,
                use_attention=True,
                use_temporal=True,
                use_waveform=True,
                dynamic_harmonics=True,
                output_samples=self.sample_rate * 4
            )
        else:
            self.model = HarmonicVAE(
                input_dim=self.input_dim,
                latent_dim=self.latent_dim,
                hidden_dims=self.hidden_dims,
                sample_rate=self.sample_rate,
                n_mels=self.n_mels
            )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Define conditions
        self.conditions = {
            "Happy": torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=device),
            "Sad": torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], device=device),
            "Energetic": torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], device=device),
            "Calm": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], device=device),
            "Classical": torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], device=device),
            "Jazz": torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], device=device),
            "Rock": torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], device=device),
            "Electronic": torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], device=device),
            "Folk": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], device=device),
            "Ambient": torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], device=device),
        }
        
        # Create output directories
        os.makedirs("output/generated", exist_ok=True)
        os.makedirs("output/reconstructed", exist_ok=True)
        os.makedirs("output/interpolated", exist_ok=True)
        
        # Setup spectrogram transforms
        self.mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=20,
            f_max=self.sample_rate//2,
            norm='slaney',
            mel_scale='htk'
        ).to(device)
        
        self.inverse_mel = InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
            f_min=20,
            f_max=self.sample_rate//2,
            mel_scale='htk'
        ).to(device)
        
        self.griffin_lim = GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            power=1.0,
            n_iter=32
        ).to(device)
    
    def preprocess_audio(self, audio_path, segment_length=4.0, n_mels=None, hop_length=None):
        """Preprocess audio file with customizable parameters"""
        if not os.path.exists(audio_path):
            raise ValueError("Audio file does not exist!")
        
        n_mels = n_mels if n_mels else self.n_mels
        hop_length = hop_length if hop_length else self.hop_length
        
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        segment_samples = int(segment_length * self.sample_rate)
        if waveform.shape[1] > segment_samples:
            start = (waveform.shape[1] - segment_samples) // 2
            waveform = waveform[:, start:start + segment_samples]
        else:
            padding = segment_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, padding))
        
        mel_transform = MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=self.sample_rate//2,
            norm='slaney',
            mel_scale='htk'
        ).to(self.device)
        
        mel_spec = mel_transform(waveform.to(self.device))
        log_mel_spec = torch.log1p(mel_spec)
        min_val, max_val = log_mel_spec.min(), log_mel_spec.max()
        if max_val > min_val:
            log_mel_spec = (log_mel_spec - min_val) / (max_val - min_val)
        
        return log_mel_spec.reshape(1, -1), waveform.to(self.device)
    
    def reconstruct_audio(self, audio_path, condition_name=None):
        """Reconstruct audio with vocoder conversion"""
        try:
            model_input, original_waveform = self.preprocess_audio(audio_path)
        except Exception as e:
            return str(e), None
        
        condition = self.conditions.get(condition_name, None)
        if condition is not None:
            condition = condition.unsqueeze(0)
        
        with torch.no_grad():
            if self.model_version == 2 and self.model.use_waveform:
                _, waveform, _, _, _ = self.model(model_input, condition)
                audio_output = waveform.squeeze().cpu().numpy()
            else:
                reconstructed, _, _, _ = self.model(model_input, condition if self.model_version == 2 else None)
                reconstructed = reconstructed.reshape(1, self.n_mels, -1)
                linear_spec = self.inverse_mel(reconstructed)
                waveform = self.griffin_lim(linear_spec)
                audio_output = waveform.squeeze().cpu().numpy()
        
        # Visualization
        orig_spec = model_input.reshape(self.n_mels, -1).cpu().numpy()
        recon_spec = reconstructed.reshape(self.n_mels, -1).cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.imshow(orig_spec, origin='lower', aspect='auto', cmap='viridis')
        ax1.set_title('Original Mel Spectrogram')
        ax2.imshow(recon_spec, origin='lower', aspect='auto', cmap='viridis')
        ax2.set_title('Reconstructed Mel Spectrogram')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        spec_image = Image.open(buf)
        
        # Save audio
        output_path = f"output/reconstructed/recon_{os.path.basename(audio_path)}"
        audio_output = audio_output / (np.max(np.abs(audio_output)) + 1e-8)
        wavfile.write(output_path, self.sample_rate, (audio_output * 32767).astype(np.int16))
        return output_path, spec_image
    
    def generate_audio(self, condition_names=None, seed=None):
        """Generate audio with multiple condition blending"""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        if condition_names and isinstance(condition_names, list):
            condition = torch.mean(torch.stack([self.conditions[c] for c in condition_names if c in self.conditions]), dim=0)
            condition = condition.unsqueeze(0)
        else:
            condition = None
        
        with torch.no_grad():
            generated = self.model.generate(num_samples=1, condition=condition)
            if self.model_version == 2 and self.model.use_waveform:
                audio_output = generated.squeeze().cpu().numpy()
            else:
                generated = generated.reshape(1, self.n_mels, -1)
                linear_spec = self.inverse_mel(generated)
                waveform = self.griffin_lim(linear_spec)
                audio_output = waveform.squeeze().cpu().numpy()
        
        # Visualization
        plt.figure(figsize=(10, 4))
        if self.model_version == 2 and self.model.use_waveform:
            plt.plot(audio_output)
            plt.title(f'Generated Waveform ({", ".join(condition_names) if condition_names else "Unconditioned"})')
        else:
            gen_spec = generated.reshape(self.n_mels, -1).cpu().numpy()
            plt.imshow(gen_spec, origin='lower', aspect='auto', cmap='viridis')
            plt.title(f'Generated Mel Spectrogram ({", ".join(condition_names) if condition_names else "Unconditioned"})')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        spec_image = Image.open(buf)
        
        output_path = f"output/generated/gen_{'_'.join(condition_names) if condition_names else 'unconditioned'}_{np.random.randint(10000)}.wav"
        audio_output = audio_output / (np.max(np.abs(audio_output)) + 1e-8)
        wavfile.write(output_path, self.sample_rate, (audio_output * 32767).astype(np.int16))
        return output_path, spec_image
    
    def interpolate_audio(self, audio_path1, audio_path2, steps=5, condition_name=None):
        """Interpolate between two audio files using slerp"""
        try:
            model_input1, _ = self.preprocess_audio(audio_path1)
            model_input2, _ = self.preprocess_audio(audio_path2)
        except Exception as e:
            return str(e), None
        
        condition = self.conditions.get(condition_name, None)
        if condition is not None:
            condition = condition.unsqueeze(0)
        
        with torch.no_grad():
            z1, _, _ = self.model.encode(model_input1, condition)
            z2, _, _ = self.model.encode(model_input2, condition)
            z1, z2 = z1.squeeze(), z2.squeeze()
            
            # Spherical linear interpolation (slerp)
            dot = torch.dot(z1, z2) / (torch.norm(z1) * torch.norm(z2))
            theta = torch.acos(dot.clamp(-1, 1))
            sin_theta = torch.sin(theta)
            if sin_theta.abs() < 1e-6:  # Avoid division by near-zero
                sin_theta = torch.tensor(1e-6, device=self.device)
            
            audio_outputs = []
            for i in range(steps):
                t = i / (steps - 1)
                slerp_z = (torch.sin((1 - t) * theta) / sin_theta * z1 + torch.sin(t * theta) / sin_theta * z2)
                slerp_z = slerp_z / torch.norm(slerp_z) * torch.norm(z1)
                slerp_z = slerp_z.unsqueeze(0)
                if self.model_version == 2 and self.model.use_waveform:
                    waveform = self.model.decode(slerp_z, condition, output_waveform=True)[1]
                    audio_outputs.append(waveform.squeeze().cpu().numpy())
                else:
                    spec = self.model.decode(slerp_z, condition)
                    spec = spec.reshape(1, self.n_mels, -1)
                    linear_spec = self.inverse_mel(spec)
                    waveform = self.griffin_lim(linear_spec)
                    audio_outputs.append(waveform.squeeze().cpu().numpy())
        
        # Visualization
        fig, axes = plt.subplots(steps, 1, figsize=(12, 3 * steps))
        if steps == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            if self.model_version == 2 and self.model.use_waveform:
                ax.plot(audio_outputs[i])
            else:
                spec = self.model.decode(slerp_z, condition).reshape(self.n_mels, -1).cpu().numpy()
                ax.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(f'Step {i+1}/{steps}')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        spec_image = Image.open(buf)
        
        audio_paths = []
        for i, audio in enumerate(audio_outputs):
            output_path = f"output/interpolated/interp_{i+1}_of_{steps}.wav"
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            wavfile.write(output_path, self.sample_rate, (audio * 32767).astype(np.int16))
            audio_paths.append(output_path)
        return audio_paths[-1], spec_image
    
    def latent_space_exploration(self, audio_path=None, x_dim=0, y_dim=1, z_dim=2, z_value=0.0, condition_name=None):
        """Interactive latent space exploration with 2D grid"""
        condition = self.conditions.get(condition_name, None)
        if condition is not None:
            condition = condition.unsqueeze(0)
        
        if audio_path:
            try:
                model_input, _ = self.preprocess_audio(audio_path)
                with torch.no_grad():
                    _, mu, _ = self.model.encode(model_input, condition)
                    base_z = mu.squeeze()
            except Exception as e:
                return str(e), None
        else:
            base_z = torch.randn(self.latent_dim, device=self.device)
        
        grid_size = 4
        x_vals = torch.linspace(-3, 3, grid_size)
        y_vals = torch.linspace(-3, 3, grid_size)
        z_grid = []
        for y in y_vals:
            for x in x_vals:
                z = base_z.clone()
                z[x_dim] = x
                z[y_dim] = y
                z[z_dim] = z_value
                z_grid.append(z)
        z_grid = torch.stack(z_grid)
        
        with torch.no_grad():
            grid_output = []
            for z in z_grid:
                z = z.unsqueeze(0)
                if self.model_version == 2 and self.model.use_waveform:
                    waveform = self.model.decode(z, condition, output_waveform=True)[1]
                    grid_output.append(waveform.squeeze().cpu().numpy())
                else:
                    spec = self.model.decode(z, condition)
                    spec = spec.reshape(1, self.n_mels, -1)
                    linear_spec = self.inverse_mel(spec)
                    waveform = self.griffin_lim(linear_spec)
                    grid_output.append(waveform.squeeze().cpu().numpy())
        
        # Visualization
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        for i, (ax, output) in enumerate(zip(axes.flatten(), grid_output)):
            y_idx = i // grid_size
            x_idx = i % grid_size
            if self.model_version == 2 and self.model.use_waveform:
                ax.plot(output)
            else:
                spec = spec.reshape(self.n_mels, -1).cpu().numpy()
                ax.imshow(spec, origin='lower', aspect='auto', cmap='viridis')
            ax.set_title(f'x={x_vals[x_idx]:.1f}, y={y_vals[y_idx]:.1f}')
            ax.axis('off')
        plt.tight_layout()
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        plt.close(fig)
        grid_image = Image.open(buf)
        
        random_idx = np.random.randint(len(grid_output))
        audio_output = grid_output[random_idx]
        output_path = f"output/generated/latent_explore_z{z_dim}_{z_value:.1f}.wav"
        audio_output = audio_output / (np.max(np.abs(audio_output)) + 1e-8)
        wavfile.write(output_path, self.sample_rate, (audio_output * 32767).astype(np.int16))
        return output_path, grid_image

def create_interface(model_path):
    """Create an enhanced Gradio interface"""
    vae_interface = HarmonicVAEInterface(model_path)
    
    with gr.Blocks(title="HarmonicVAE 2.0 Interactive Interface") as interface:
        gr.Markdown("# HarmonicVAE 2.0 Interactive Interface")
        gr.Markdown("Explore music generation with a psychoacoustically-enhanced VAE. [Learn More](#)")
        
        with gr.Tab("Audio Generation"):
            with gr.Row():
                with gr.Column():
                    condition = gr.Dropdown(list(vae_interface.conditions.keys()) + ["None"], label="Condition", value="None", multiselect=True)
                    seed = gr.Number(label="Seed (optional)", value=None)
                    generate_btn = gr.Button("Generate")
                with gr.Column():
                    audio_out = gr.Audio(label="Generated Audio")
                    image_out = gr.Image(label="Visualization")
            generate_btn.click(
                lambda c, s: vae_interface.generate_audio(c if c != ["None"] else None, s),
                [condition, seed], [audio_out, image_out]
            )
        
        with gr.Tab("Audio Reconstruction"):
            with gr.Row():
                with gr.Column():
                    audio_in = gr.Audio(label="Input Audio", type="filepath")
                    recon_cond = gr.Dropdown(list(vae_interface.conditions.keys()) + ["None"], label="Condition", value="None")
                    recon_btn = gr.Button("Reconstruct")
                with gr.Column():
                    recon_audio = gr.Audio(label="Reconstructed Audio")
                    recon_image = gr.Image(label="Spectrogram Comparison")
            recon_btn.click(
                lambda a, c: vae_interface.reconstruct_audio(a, None if c == "None" else c),
                [audio_in, recon_cond], [recon_audio, recon_image]
            )
        
        with gr.Tab("Audio Interpolation"):
            with gr.Row():
                with gr.Column():
                    audio1 = gr.Audio(label="Start Audio", type="filepath")
                    audio2 = gr.Audio(label="End Audio", type="filepath")
                    steps = gr.Slider(3, 10, value=5, step=1, label="Steps")
                    interp_cond = gr.Dropdown(list(vae_interface.conditions.keys()) + ["None"], label="Condition", value="None")
                    interp_btn = gr.Button("Interpolate")
                with gr.Column():
                    interp_audio = gr.Audio(label="Last Interpolated Audio")
                    interp_image = gr.Image(label="Interpolation Visualization")
            interp_btn.click(
                lambda a1, a2, s, c: vae_interface.interpolate_audio(a1, a2, s, None if c == "None" else c),
                [audio1, audio2, steps, interp_cond], [interp_audio, interp_image]
            )
        
        with gr.Tab("Latent Space Exploration"):
            with gr.Row():
                with gr.Column():
                    latent_audio = gr.Audio(label="Starting Audio (optional)", type="filepath")
                    x_dim = gr.Slider(0, vae_interface.latent_dim-1, value=0, step=1, label="X Dimension")
                    y_dim = gr.Slider(0, vae_interface.latent_dim-1, value=1, step=1, label="Y Dimension")
                    z_dim = gr.Slider(0, vae_interface.latent_dim-1, value=2, step=1, label="Fixed Z Dimension")
                    z_val = gr.Slider(-3, 3, value=0, step=0.1, label="Z Value")
                    explore_cond = gr.Dropdown(list(vae_interface.conditions.keys()) + ["None"], label="Condition", value="None")
                    explore_btn = gr.Button("Explore")
                with gr.Column():
                    explore_audio = gr.Audio(label="Sample from Grid")
                    explore_image = gr.Image(label="Latent Space Grid")
            explore_btn.click(
                lambda a, x, y, z, z_v, c: vae_interface.latent_space_exploration(a, int(x), int(y), int(z), z_v, None if c == "None" else c),
                [latent_audio, x_dim, y_dim, z_dim, z_val, explore_cond], [explore_audio, explore_image]
            )
    
    return interface

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="HarmonicVAE Interactive Interface")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    interface = create_interface(args.model)
    interface.launch(server_port=args.port, share=args.share, max_threads=10)
