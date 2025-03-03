Below is the comprehensive **HarmonicVAE Training Manual**, designed to guide you through the process of training a Variational Autoencoder tailored for music generation. This manual combines technical precision with a dash of humor to keep you engaged while you teach a neural network to groove better than most humans. Whether you're aiming to craft the next hit single or just curious about blending genres in unexpected ways, this guide has you covered.

---

# HarmonicVAE Training Manual

*So, you want to teach a neural network to understand music better than most humans? Buckle up, friend.*

This guide will take you from setup to musical mastery with HarmonicVAE, a specialized Variational Autoencoder that leverages psychoacoustic principles and music theory. Whether you're a machine learning enthusiast or an audio tinkerer, you'll find everything you need here to get started, troubleshoot, and even push the boundaries of generative audio.

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

---

## Prerequisites

Before you dive into this musical adventure, ensure you have:

- **Hardware**: A GPU (NVIDIA with 8GB+ VRAM recommended). Training on a CPU is possible, but you might grow a beard waiting for it to finish.
- **Software**: Python 3.7+, PyTorch 1.7+, and dependencies listed in `requirements.txt`.
- **Data**: Access to datasets like MAESTRO, GTZAN, or your own audio files.
- **Knowledge**: Basic Python and PyTorch skills. Music theory or audio processing experience is a bonus but not required.
- **Patience**: Neural networks learn at their own pace—sometimes it’s a waltz, sometimes it’s free jazz.

---

## Environment Setup

Setting up your environment is the first step to harmonic success. Here are two reliable methods:

### The Bulletproof Setup Method

```bash
# Create a virtual environment
python -m venv harmonic_env

# Activate it (platform-specific)
# Windows:
harmonic_env\Scripts\activate
# macOS/Linux:
source harmonic_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Check GPU availability
python -c "import torch; print('GPU Available:', torch.cuda.is_available(), '\nDevice:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Docker Alternative

For container fans:

```bash
# Build the Docker image
docker build -t harmonic-vae .

# Run with GPU support and mounted directories
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output harmonic-vae
```

**Tip**: If the GPU check fails, ensure CUDA and cuDNN are installed correctly. Training sans GPU is like composing a symphony with a kazoo—possible, but painful.

---

## Dataset Preparation

HarmonicVAE thrives on audio data. Here’s how to prepare it:

### Supported Datasets

1. **MAESTRO**: Classical piano performances (~80GB).
2. **GTZAN**: Multi-genre tracks (~1.2GB).
3. **Custom**: Your personal audio stash.

### Dataset Download

```bash
# MAESTRO (big download, grab a coffee)
python -m src.harmonic_audio_integration --dataset maestro --dataset_dir data/maestro --mode download

# GTZAN (quicker, less storage)
python -m src.harmonic_audio_integration --dataset gtzan --dataset_dir data/gtzan --mode download
```

### Custom Dataset Guidelines

Using your own tunes? Follow these steps:

1. **Format**: WAV, MP3, FLAC, OGG, or M4A.
2. **Duration**: 3-10 seconds per sample (longer files get chopped).
3. **Organization**: Sort into subdirectories (e.g., `genre/artist/album`) for metadata ease.
4. **Preprocessing**:
   ```bash
   python -m src.harmonic_audio_integration --dataset custom --custom_dir path/to/audio --mode preprocess --output_dir data/processed_custom
   ```

**Pro Tip**: Quality beats quantity. Feed it 500 diverse, crisp tracks over 5,000 muddy ones—your model will thank you with better tunes.

---

## Training Configuration

Configure HarmonicVAE to suit your needs. Here’s the rundown:

### Key Parameters

| Parameter       | Description                                      | Recommended Values       |
|-----------------|--------------------------------------------------|--------------------------|
| `input_dim`     | Input size (auto-calculated from audio params)   | N/A                      |
| `latent_dim`    | Latent space size                                | 64-256 (128 default)     |
| `hidden_dims`   | Encoder/decoder layer sizes                      | [1024, 512, 256] or simpler [512, 256] |
| `learning_rate` | Gradient step size                               | 0.001-0.0001             |
| `batch_size`    | Samples per update                               | 16-64 (GPU-dependent)    |
| `epochs`        | Dataset passes                                   | 50-200                   |
| `beta`          | KL divergence weight                             | 0.0001 to 0.01 (anneal)  |

### Configuration Examples

#### Quick Test Run

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 20 --batch_size 16 --latent_dim 64 --output_dir output/quick_test
```

#### High-Quality Setup

```bash
python -m src.harmonic_audio_integration --dataset maestro --mode train --epochs 100 --batch_size 32 --latent_dim 256 --output_dir output/maestro_full
```

#### Configuration File (Recommended)

Create `config.json`:

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

Run it:

```bash
python -m src.harmonic_audio_integration --config config.json
```

---

## Training Process

Time to train! Here’s how to kick things off:

### Basic Training

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan
```

### With Validation

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --validation_split 0.2
```

### Resume from Checkpoint

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --resume_from output/gtzan/harmonic_vae_epoch_30.pt
```

### Training Stages

1. **Initial Phase (Epochs 1-10)**  
   - High loss, messy latent space.  
   - Goal: Confirm it’s running.  
   - Vibe: “Did I break something already?”

2. **Middle Phase (Epochs 11-50)**  
   - Reconstruction improves, KL ramps up.  
   - Goal: Balance losses.  
   - Vibe: “Okay, we’re getting somewhere!”

3. **Late Phase (Epochs 51+)**  
   - Fine-tuning kicks in.  
   - Goal: Avoid overfitting, shape latent space.  
   - Vibe: “When can I hear something cool?”

---

## Monitoring and Evaluation

Keep tabs on your model’s progress:

### Real-Time Logs

Expect output like:

```
Epoch 10/50, Train Loss: 0.0456, Val Loss: 0.0478, Recon: 0.0389, KL: 0.0089, LR: 0.00095
```

### TensorBoard

Visualize with:

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan --use_tensorboard True
tensorboard --logdir=output/gtzan/logs
```

Visit `http://localhost:6006`.

### Visualizations

Every 10 epochs, check `output_dir/visualizations/` for:
- Reconstructions.
- Latent space t-SNE plots.
- Novel samples.
- Mel spectrograms.

### Metrics

- **Reconstruction Loss**: Measures input fidelity.
- **KL Divergence**: Ensures latent space regularity.
- **Sparsity**: Higher = more efficient latent use.
- **Spectrograms**: Visual audio quality check.
- **Listen Up**: Metrics don’t sing—trust your ears!

---

## Hyperparameter Tuning

Tweak these for optimal results:

### Key Hyperparameters

- **latent_dim**: Bigger = more creativity, less order.
- **hidden_dims**: Deeper = complex, slower training.
- **learning_rate**: High = risky, low = sluggish.
- **batch_size**: Bigger = stable, memory-hungry.
- **beta**: Reconstruction vs. latent structure tradeoff.

### Tuning Approach

1. **Baseline**: Train with defaults.
2. **Sweep**: Test `latent_dim` (64, 128, 256), `beta` (0.001, 0.01, 0.1).
3. **Refine**: Adjust `hidden_dims`, then `learning_rate`.

### Optuna Automation

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode tune --trials 50 --output_dir output/tuning
```

---

## Advanced Techniques

Level up your training:

### Beta Annealing

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --beta_start 0.0001 --beta_end 0.01 --beta_steps 20 --output_dir output/gtzan_annealing
```

### Data Augmentation

```bash
python -m src.harmonic_audio_integration --dataset gtzan --mode train --augment True --augment_strength 0.5 --output_dir output/gtzan_augmented
```

Includes pitch shifts, time stretching, and noise.

### Multi-Dataset Training

```bash
python -m src.harmonic_audio_integration --dataset combined --maestro_dir data/maestro --gtzan_dir data/gtzan --custom_dir data/custom --mode train --output_dir output/combined
```

### Transfer Learning

```bash
# Base model
python -m src.harmonic_audio_integration --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan_base
# Fine-tune
python -m src.harmonic_audio_integration --dataset custom --custom_dir path/to/audio --mode train --epochs 20 --resume_from output/gtzan_base/harmonic_vae_best.pt --output_dir output/custom_finetuned
```

---

## Troubleshooting

### Common Errors

| Error                   | Cause                          | Fix                                      |
|-------------------------|-------------------------------|------------------------------------------|
| `CUDA out of memory`    | Too much GPU load             | Lower `batch_size` or `hidden_dims`      |
| `No valid audio files`  | Bad formats or short clips    | Validate with `--mode validate_data`     |
| `Loss is NaN`           | Unstable training             | Reduce `learning_rate`, add clipping     |
| `Noisy output`          | High beta or short training   | Lower `beta`, train longer               |

### Performance Fixes

- **Slow Training**: Use `--mixed_precision True`, shrink model.
- **Poor Audio**: Increase resolution (`--n_mels 256 --n_fft 4096`), check preprocessing.
- **Memory Issues**: Reduce `batch_size`, use `--grad_accum_steps 4`.

**Debugging Mantra**: Restart, recheck, reread—then resolve.

---

## Case Studies

### GTZAN (Genre Mix)

- **Setup**: 10 genres, 3-second clips.
- **Config**: `latent_dim=128`, `epochs=100`, `hidden_dims=[1024, 512, 256]`.
- **Results**: Genre clusters by epoch 60, smooth interpolations.

### MAESTRO (Piano)

- **Setup**: Piano, 4-second clips.
- **Config**: `latent_dim=256`, `epochs=100`, deeper `hidden_dims`.
- **Results**: Structured outputs by epoch 80, dynamic latent space.

---

## Final Tips

- Start small, scale up.
- Listen to outputs often—trust your ears.
- Name runs descriptively (e.g., `gtzan_ld128_e100`).
- Save checkpoints religiously.
- Experiment wildly in the latent space.

Happy training! May your HarmonicVAE sing beautifully—or at least surprise you pleasantly.

---

*"The beautiful thing about learning is that nobody can take it away from you."* — B.B. King

*"The beautiful thing about neural networks is sometimes they learn what you intended."* — Anonymous ML Practitioner
