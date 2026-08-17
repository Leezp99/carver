#!/usr/bin/env python3
"""Carver reconstruction CLI: waveform in -> reconstructed waveform out.

Examples
--------
# Adaptive (auto) mode, near-lossless:
python reconstruct.py --ckpt checkpoints/carver --input in.wav --output out.wav --rate auto

# Full frame rate (upper bound):
python reconstruct.py --ckpt checkpoints/carver --input in.wav --output out.wav --rate 1.0

# Fixed keep-ratio (e.g. keep 70% of frames):
python reconstruct.py --ckpt checkpoints/carver --input in.wav --output out.wav --rate 0.7
"""

import argparse
from pathlib import Path

import torch

from carver import load_model, load_audio, reconstruct, save_audio


def parse_rate(v: str):
    return v if v == "auto" else float(v)


def main():
    ap = argparse.ArgumentParser(description="Carver variable-rate speech VAE reconstruction")
    ap.add_argument("--ckpt", required=True, help="checkpoint dir containing weights.pth + metadata.pth")
    ap.add_argument("--input", required=True, help="input audio file (any sample rate / channels)")
    ap.add_argument("--output", required=True, help="output .wav path")
    ap.add_argument("--rate", type=parse_rate, default="auto",
                    help="'auto' (near-lossless), 1.0 (full rate), or a keep-ratio in (0,1)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.device)
    audio = load_audio(args.input, args.device)
    wav, info = reconstruct(model, audio, args.rate)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_audio(wav, args.output)
    print(f"[carver] rate={info['requested_rate']:.4f} "
          f"kept={info['mask_rate']*100:.1f}%  ->  {args.output}")


if __name__ == "__main__":
    main()
