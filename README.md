# Carver: Content-Adaptive Variable-Rate Speech VAE via Reconstruction-Error Routing

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://github.com/leezp99/carver)
[![Demo](https://img.shields.io/badge/Demo-Listen-1f6feb?logo=github&logoColor=white)](https://leezp99.github.io/carver-demo/)
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-ffce1c?logo=huggingface&logoColor=white)](https://huggingface.co/leezp99/carver)

**Carver** is a single **continuous, 24 kHz, 25 Hz, 64-dim** speech VAE that turns one
set of weights into a whole **frame-rate–distortion curve**. It scores every latent
frame by its **waveform reconstruction error** and keeps or drops it accordingly, so
the same model runs at full rate, at an adaptive near-lossless **auto** rate, or at
any fixed keep-ratio without retraining. The [code repository](https://github.com/leezp99/carver)
provides the official **inference (reconstruction)** code and these pretrained weights.

🔊 Listen to reconstructions on the [demo page](https://leezp99.github.io/carver-demo/).

## ✨ Highlights

- **One model, many rates** — a multi-rate masking curriculum lets a single checkpoint
cover a continuous frame-rate–distortion trade-off, with no per-rate retraining.
- **Reconstruction-error routing** — a stop-gradient full-frame decode scores each
frame by its waveform L1 error; the hardest-to-reconstruct frames are kept.
- **Adaptive `auto` mode** — a per-utterance near-lossless budget that drops mostly
silence, plus explicit control at any fixed keep-ratio.
- **Continuous latent** — a compact 25 Hz / 64-dim latent, a convenient front-end for
downstream diffusion / autoregressive speech generators.

## 🛠️ Installation

Tested on Ubuntu with **Python 3.10** and **CUDA 12.x** (CPU also works).

```bash
git clone https://github.com/leezp99/carver.git
cd carver

conda create -n carver python=3.10 -y
conda activate carver

pip install -r requirements.txt
```

## 📦 Pretrained models

| Model  | Sample rate | Latent         | Variable rate           | Download                                                    |
| ------ | ----------- | -------------- | ----------------------- | ----------------------------------------------------------- |
| Carver | 24 kHz      | 25 Hz / 64-dim | `auto` + any fixed rate | [🤗 Hugging Face](https://huggingface.co/leezp99/carver)    |

A checkpoint is a directory with **two files that must live together**:

```
checkpoints/carver/
├── weights.pth      # model weights (~1.4 GB)
└── metadata.pth     # constructor kwargs (small; defines the architecture)
```

`metadata.pth` carries the exact constructor arguments used to build the model, so the
architecture always matches the weights (no config editing needed). Download with:

```bash
pip install -U huggingface_hub
hf download leezp99/carver --local-dir checkpoints/carver
```

## 🚀 Usage

### Reconstruction via CLI

```bash
# Adaptive near-lossless reconstruction:
python reconstruct.py --ckpt checkpoints/carver --input assets/example.wav --output out_auto.wav --rate auto

# Full frame rate (model's upper bound):
python reconstruct.py --ckpt checkpoints/carver --input assets/example.wav --output out_full.wav --rate 1.0

# Fixed keep-ratio, e.g. keep 70% of frames:
python reconstruct.py --ckpt checkpoints/carver --input assets/example.wav --output out_070.wav --rate 0.7
```

| Argument   | Description                                                              |
| ---------- | ------------------------------------------------------------------------ |
| `--ckpt`   | Checkpoint dir containing `weights.pth` + `metadata.pth`.                |
| `--input`  | Input audio path (any sample rate / channels; resampled to 24 kHz mono). |
| `--output` | Output `.wav` path.                                                      |
| `--rate`   | `auto` (near-lossless), `1.0` (full rate), or a keep-ratio in `(0, 1)`.  |
| `--device` | `cuda` (default if available) or `cpu`.                                  |

### Reconstruction in Python

```python
import torch
from carver import load_model, load_audio, reconstruct, save_audio

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("checkpoints/carver", device)

audio = load_audio("assets/example.wav", device)    # mono, 24 kHz, -16 LUFS -> [1, 1, L]
wav, info = reconstruct(model, audio, rate="auto")  # rate: "auto" | 1.0 | 0.7 | ...
print(info)                                         # {'requested_rate': 0.67, 'mask_rate': 0.67}
save_audio(wav, "out.wav")
```

## ❤️ Acknowledgements

We gratefully build on these excellent open-source works, reusing and adapting a good deal of their code:

- InfoTok
- Descript Audio Codec (DAC)
- Fish-Speech

## 📄 License

Released under the [MIT License](https://github.com/leezp99/carver/blob/main/LICENSE).
