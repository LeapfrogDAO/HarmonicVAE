# HarmonicVAE Training Manual

*So you want to teach a neural network to understand music better than most humans? Buckle up, friend.*

This comprehensive guide will walk you through training your very own HarmonicVAE from scratch to musical mastery. Whether you're looking to generate the next chart-topper or just want to see what happens when you blend death metal with bossa nova (spoiler: it's weird), this manual has got you covered.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Dataset Preparation](#dataset-preparation)
4. [Training Configuration](#training-configuration)
5. [Training Process](#training-process)
6. [Monitoring and Evaluation](#monitoring-and-evaluation)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Advanced Techniques](#advanced-techniques)
9. [Troubleshooting](#troubleshooting)
10. [Case Studies](#case-studies)

## Prerequisites

Before you embark on this musical journey, make sure you've got:

- **Hardware**: A decent GPU (NVIDIA with 8GB+ VRAM recommended). Yes, you *could* train this on CPU, but you'll age considerably in the process.
- **Software**: Python 3.7+, PyTorch 1.7+, and other dependencies listed in `requirements.txt`
- **Data**: Access to audio datasets (MAESTRO, GTZAN, or your personal collection)
- **Patience**: Neural nets are like toddlers learning an instrument—adorable but frustratingly slow to improve

## Environment Setup

### The Bulletproof Setup Method

```bash
# Create a fresh virtual environment (isolation is your friend)
python -m venv harmonic_env

# Activate it (platform-specific)
# On Windows:
harmonic_env\Scripts\activate
# On macOS/Linux:
source harmonic_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify GPU access if applicable
python -c "import torch; print('GPU Available:', torch.cuda.is_available(), '\nDevice:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Docker Alternative

For the containerization enthusiasts:

```bash
# Build the Docker image
docker build -t harmonic-vae .

# Run with GPU support
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output harmonic-vae
```

## Dataset Preparation

### Supported Datasets

1. **MAESTRO**: Piano performances (classical)
2. **GTZAN**: Multi-genre classification dataset
3. **Custom**: Your own audio files

### Dataset Download

```bash
# For MAESTRO (warning: ~80GB)
python -m src.harmonic_audio_integration --dataset maestro --dataset_dir data/maestro --mode download

# For GTZAN (~1.2GB)
python -m src.harmonic_audio_integration --dataset gtzan --dataset_dir data/gtzan --mode download
```

### Custom Dataset Guidelines

If you're using your own audio collection:

1. **Format**: WAV, MP3, FLAC, OGG, or M4A files
2. **Duration**: Ideally 3-10 seconds per sample (longer files will be segmented)
3. **Organization**: Organizing by genre/artist/album in subdirectories helps with metadata
4. **Preprocessing**: 
   ```bash
   # Normalize and check your custom dataset
   python -m src.harmonic_audio_integration --dataset custom --custom_dir path/to/audio --mode preprocess --output_dir data/processed_custom
   ```

**Pro Tip**: *Quality trumps quantity.* 500 well-produced, diverse tracks will yield better results than 5,000 similar-sounding, low-quality samples. Your model will only be as good as the audio you feed it!

## Training Configuration

### Key Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `input_dim` | Dimension of the input (depends on audio params) | Calculated automatically |
| `latent_dim` | Size of the latent space | 64-256 (128 default) |
| `hidden_dims` | Architecture of encoder/decoder | [1024, 512, 256] for deep, [512, 256] for faster training |
| `learning_rate` | Controls step size during training | 0.001-0.0001 |
| `batch_size` | Number of samples per gradient update | 16-64 (GPU dependent) |
| `epochs` | Full passes through the dataset | 50-200 (more for larger datasets) |
| `beta` | KL divergence weight (controls latent space) | Start low (0.0001) and anneal to 0.01 |

### Configuration Examples

#### Quick Training (Testing Setup)
```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 20 --batch_size 16 --latent_dim 64 --output_dir output/quick_test
```

#### High-Quality Training
```bash
python -m src.harmonic_audio_integration --dataset maestro --mode train --epochs 100 --batch_size 32 --latent_dim 256 --output_dir output/maestro_full
```

#### Create a Configuration File (Recommended)
```json
{
    "dataset": "gtzan",
    "dataset_dir": "data/gtzan",
    "mode": "train",
    "epochs": 100,
    "batch_size": 32,
    "latent_dim": 128,
    "hidden_dims": [1024, 512, 256],
    "learning_rate": 0.0005,
    "beta_start": 0.0001,
    "beta_end": 0.01,
    "segment_length": 3.0,
    "output_dir": "output/gtzan_production",
    "checkpoint_every": 10
}
```

Save as `config.json` and run:
```bash
python -m src.harmonic_audio_integration --config config.json
```

## Training Process

### Basic Training

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan
```

### Training with Validation

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --validation_split 0.2
```

### Resume Training From Checkpoint

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --resume_from output/gtzan/harmonic_vae_epoch_30.pt
```

### Training Stages

1. **Initial Phase (Epochs 1-10)**
   * High reconstruction loss, chaotic latent space
   * Focus: Just making sure everything's working
   * Mood: "What have I done?"

2. **Middle Phase (Epochs 11-50)**
   * Reconstruction improves, KL divergence increases
   * Focus: Balance reconstruction vs. KL loss
   * Mood: "Hey, this might actually work!"

3. **Late Phase (Epochs 51+)**
   * Fine-tuning, optimization, regularization
   * Focus: Preventing overfitting, latent space structure
   * Mood: "Is it done yet? I need to touch grass."

## Monitoring and Evaluation

### Real-time Monitoring

The training script outputs metrics for each epoch:
```
Epoch 10/50, Train Loss: 0.0456, Val Loss: 0.0478, Recon: 0.0389, KL: 0.0089, LR: 0.00095
```

### TensorBoard Integration

For those who like pretty graphs:

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --use_tensorboard True
```

Then:
```bash
tensorboard --logdir=output/gtzan/logs
```

Navigate to `http://localhost:6006` in your browser.

### Visualization During Training

Every 10 epochs (configurable), the training process will:
1. Generate visualization of reconstructions
2. Create latent space t-SNE plots
3. Generate novel samples
4. Save mel spectrograms of the results

Look for these in your output directory under `visualizations/`.

### Evaluation Metrics

Quantitative evaluation is tricky for generative audio. We use:

1. **Reconstruction Loss**: How well the VAE can reconstruct inputs
2. **KL Divergence**: How well the latent space matches a standard normal distribution
3. **Sparsity**: Percentage of near-zero activations in the latent space (higher is better)
4. **Spectrograms**: Visual comparison of reconstructions
5. **Latent Space Structure**: t-SNE or UMAP visualizations for cluster analysis

**Warning**: *Low loss does not always equal good music. **Always** listen to the outputs!*

## Hyperparameter Tuning

### Key Hyperparameters to Tune

#### Architectural Parameters
- **latent_dim**: Larger values = more expressive but less structured latent space
- **hidden_dims**: Deeper networks can capture more complex relationships but train slower

#### Training Parameters
- **learning_rate**: Too high = unstable training, too low = painfully slow training
- **batch_size**: Larger batches = more stable gradients but higher memory usage
- **beta**: Controls latent space structure vs. reconstruction quality

### Automated Tuning with Optuna

For the really obsessive (we see you):

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode tune --trials 50 --output_dir output/tuning
```

This will run Optuna to search for optimal hyperparameters, which is the deep learning equivalent of having a personal trainer who makes you try 50 different workout routines.

### Practical Tuning Schedule

1. **Start simple**: Train a baseline model with default parameters
2. **Latent dimension sweep**: Try 64, 128, 256 (holding other params constant)
3. **Beta sweep**: Try 0.001, 0.01, 0.1 for fixed beta (or adjust annealing schedule)
4. **Architecture test**: Compare different hidden_dims configurations
5. **Learning rate optimization**: Fine-tune learning rate last

## Advanced Techniques

### Beta Annealing

Start with low beta value and gradually increase:

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --beta_start 0.0001 --beta_end 0.01 --beta_steps 20 --output_dir output/gtzan_annealing
```

### Data Augmentation

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --augment True --augment_strength 0.5 --output_dir output/gtzan_augmented
```

Augmentations include:
- Pitch shifting (±2 semitones)
- Time stretching (±20%)
- Random gain adjustment
- Background noise addition

### Multi-dataset Training

```bash
python -m src.harmonic_audio_integration --dataset combined --maestro_dir data/maestro --gtzan_dir data/gtzan --custom_dir data/custom --mode train --output_dir output/combined
```

### Transfer Learning

```bash
# First train on GTZAN
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan_base

# Then fine-tune on your custom dataset
python -m src.harmonic_audio_integration --dataset custom --custom_dir path/to/audio --mode train --epochs 20 --resume_from output/gtzan_base/harmonic_vae_best.pt --output_dir output/custom_finetuned
```

## Troubleshooting

### Common Error Messages

| Error | Probable Cause | Solution |
|-------|---------------|----------|
| `CUDA out of memory` | Batch size too large or model too big | Reduce batch_size or hidden_dims |
| `No valid audio files found` | Audio format issues or files too short | Check files with `--mode validate_data` |
| `Loss is NaN` | Learning rate too high or gradient explosion | Lower learning rate, add gradient clipping with `--clip_grad` |
| `Reconstruction is all noise` | Training unstable or beta too high | Lower beta value, increase training time |
| `ImportError: No module named src` | Running from wrong directory | Run from project root with `-m src.harmonic_audio_integration` |

### Performance Issues

1. **Training too slow**: 
   - Enable mixed precision with `--mixed_precision True`
   - Reduce hidden_dims or batch_size
   - Check if CUDA is actually being used

2. **Audio quality poor**: 
   - Increase input resolution with `--n_mels 256 --n_fft 4096`
   - Check preprocessing with `--mode visualize_preprocessing`
   - Lower beta value to prioritize reconstruction

3. **Out of memory**:
   - Reduce batch size
   - Use gradient accumulation with `--grad_accum_steps 4`
   - Move to smaller hidden_dims

### When All Else Fails

The five stages of debugging:
1. **Denial**: "My code is perfect, it must be the hardware."
2. **Anger**: *Keyboard percussion solo*
3. **Bargaining**: "Maybe if I restart everything..."
4. **Depression**: *Stares at loss curve that refuses to decrease*
5. **Acceptance**: *Actually reads documentation and carefully debugs step by step*

Take heart: neural networks are stubborn creatures by nature.

## Case Studies

### Case Study 1: Genre Classification Dataset (GTZAN)

**Setup**:
- 10 genres, 100 tracks per genre
- 3-second segments, 22.05 kHz sample rate
- 128 mel bands, 2048 FFT window

**Training Configuration**:
```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 100 --batch_size 32 --latent_dim 128 --hidden_dims 1024 512 256 --learning_rate 0.0005 --beta_start 0.0001 --beta_end 0.01 --output_dir output/gtzan_full
```

**Results**:
- Convergence around epoch 60
- Clear genre clustering in latent space
- Most confusion between rock/metal and jazz/classical
- Best generations from classical and jazz (more structured)
- Interpolation between genres creates interesting transitions

**Listening Examples**: Find examples in `output/gtzan_full/examples/`

### Case Study 2: Piano Music (MAESTRO)

**Setup**:
- Piano performances across styles/composers
- 4-second segments, 22.05 kHz sample rate
- 128 mel bands, 2048 FFT window

**Training Configuration**:
```bash
python -m src.harmonic_audio_integration --dataset maestro --mode train --epochs 100 --batch_size 24 --latent_dim 256 --hidden_dims 1024 512 256 128 --learning_rate 0.0003 --beta_start 0.0001 --beta_end 0.005 --output_dir output/maestro_full
```

**Results**:
- Slower convergence (around epoch 80)
- Latent space organized by playing style and dynamics
- High-quality piano reconstructions
- Novel generations maintain musical structure
- Interesting discoveries in latent space traversal

**Note**: Piano benefits from longer segments (4+ seconds) to capture phrasing

---

## Final Tips for Success

1. **Start small, then scale up**: Begin with a small subset of data and simple model
2. **Listen constantly**: Numbers lie, ears don't. Regularly generate and listen to samples
3. **Track experiments**: Name output dirs with hyperparameters (e.g., `gtzan_ld128_e100_b0.01`)
4. **Backup checkpoints**: Save intermediate models in case of power failure/crashes
5. **Explore the latent space**: The most interesting results come from creative sampling

Remember, training a HarmonicVAE isn't just about achieving low loss values—it's about teaching a machine to understand and generate something deeply human. Be patient, experiment boldly, and most importantly, have fun listening to your AI's musical journey from random noise to (hopefully) musical brilliance.

Happy training! May your loss curves be monotonically decreasing and your generated music surprisingly listenable.

---

*"The beautiful thing about learning is that nobody can take it away from you."* — B.B. King

*"The beautiful thing about neural networks is sometimes they learn what you intended."* — Anonymous ML Practitioner
