# HarmonicVAE

Welcome to the **HarmonicVAE (VAE)**—a neural network that’s basically a musical genius trapped in code. This bad boy doesn’t just crunch audio; it *gets* music, weaving psychoacoustics and music theory into its DNA. It’s like giving a VAE a conservatory degree and a backstage pass to real-world datasets like MAESTRO and GTZAN. Let’s make some noise—literally!

## Features

- **Psychoacoustic Modeling**: Mel spectrograms and harmonic vibes straight out of music theory, because your ears deserve better than flat frequencies.
- **Variational Architecture**: A latent space so smooth you could slide between musical ideas like a DJ on a sugar high.
- **Hierarchical Representations**: Catches everything from subtle timbres to epic musical arcs—like a composer with X-ray hearing.
- **LLM Bridge**: Optional hookup to Large Language Models, so you can whisper “make me a banger” and watch it deliver.
- **Real Audio Integration**: Trains on MAESTRO (piano perfection) and GTZAN (genre galore), plus your own audio stash. No more fake beats here!

## Requirements

- **Python 3.7+**: We’re not stuck in the Dark Ages.
- **PyTorch 1.7+**: The engine under the hood.
- **torchaudio**: Audio magic, unleashed.
- **numpy**: For when the numbers need wrangling.
- **matplotlib**: Pretty graphs to impress your friends.
- **scikit-learn**: Data splitting like a ninja.
- **(Optional) transformers**: For that sweet text-to-tune action.
- **(Optional) plotly**: Fancy interactive visuals that scream “look at me!”

## Installation

```bash
# Snag the repo like it’s hot
git clone https://github.com/yourusername/harmonic-vae.git
cd harmonic-vae

# Load up the goods
pip install -r requirements.txt
```

## Quick Start

### Training on GTZAN Dataset

```bash
python integrate_harmonic_vae.py --dataset gtzan --mode train --epochs 50 --output_dir output/gtzan
```
*Unleash the model on GTZAN and watch it vibe from blues to metal like a genre-hopping rockstar.*

### Training on MAESTRO Dataset

```bash
python integrate_harmonic_vae.py --dataset maestro --mode train --epochs 50 --output_dir output/maestro
```
*Let it soak up MAESTRO’s piano brilliance—Chopin would approve (maybe).*

### Training on Custom Audio Files

```bash
python integrate_harmonic_vae.py --dataset custom --custom_dir path/to/audio/files --mode train --epochs 50 --output_dir output/custom
```
*Got a secret audio hoard? Dump it in and see what this beast cooks up. No judgment, even if it’s all polka.*

### Generating New Audio from a Trained Model

```bash
python integrate_harmonic_vae.py --mode generate --model_path output/gtzan/harmonic_vae_best.pt --output_dir output/generated
```
*Crank out fresh tunes. Will it be a hit or a hilarious flop? Spin the wheel!*

### Visualizing the Latent Space of Your Audio Files

```bash
python integrate_harmonic_vae.py --mode visualize --model_path output/gtzan/harmonic_vae_best.pt --custom_dir path/to/audio/files --output_dir output/visualizations
```
*Peek into the latent space—it’s like a cosmic map of your music collection. Zoom in, get lost, have fun.*

### Interpolating Between Two Audio Files

```bash
python integrate_harmonic_vae.py --mode interpolate --model_path output/gtzan/harmonic_vae_best.pt --audio1 path/to/audio1.wav --audio2 path/to/audio2.wav --output_dir output/interpolations
```
*Mix two tracks like a sonic smoothie blender. Jazz meets dubstep? Let’s find out!*

## Architecture Details

This Harmonic VAE is a musical marvel built from the ground up:

1. **Audio Processing**: Turns raw audio into mel spectrograms—because humans hear fancy, not flat.
2. **Encoder Network**: Squashes audio into a latent nugget of pure musical essence.
3. **Harmonic Layers**: Special sauce layers that know a chord from a chaos, thanks to music theory.
4. **Latent Space**: A playground where musical ideas hang out, ready to mingle.
5. **Decoder Network**: Spins latent dreams back into sound waves you can actually hear.
6. **LLM Bridge**: Optional gadget to chat with language models—because “sad violin” should mean something.

## Example Results

Once trained, this model can:

- **Generate novel jams**: New music from scratch—maybe a masterpiece, maybe a meme.
- **Interpolate like a pro**: Blend genres or pieces smoother than a jazz sax solo.
- **Visualize the magic**: Plot your audio in 2D and see the family resemblance between tracks.
- **Text-to-tune**: With the LLM bridge, turn “epic battle theme” into an actual epic battle theme.

## Extending the Model

This isn’t a one-trick pony—tweak it to your heart’s content:

- **Attention Mechanisms**: Add some focus for those long, winding musical tales.
- **Conditional Generation**: Make it churn out “happy techno” or “stormy blues” on command.
- **Transformer Integration**: Go full sci-fi with text-to-music transformers.
- **Waveform Modeling**: Ditch spectrograms and go raw with waveforms for that gritty edge.

## Citation

If you use this to wow the world (or just your cat), give a shoutout:

```
@software{harmonic_vae,
  author = {Your Name},
  title = {Harmonic Autoencoder for Music},
  year = {2025},
  url = {https://github.com/yourusername/harmonic-vae}
}
```

## License

The Apache License 2.0 strikes the perfect balance for HarmonicVAE: good vibes should be shared

## Acknowledgements

Big props to the wizards of neural audio, music info retrieval, and psychoacoustics research. Extra love to the open-source crew and the sacred bean juice (coffee) that fueled this madness.

---

There you go! This README keeps all the technical goodness, adds the latest features, and sprinkles in humor to make it a fun read. Feel free to tweak the sass level or add your own flair—enjoy your musical coding adventure!
