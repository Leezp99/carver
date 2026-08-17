"""Carver inference helpers: load a checkpoint and reconstruct a waveform."""

from pathlib import Path
from typing import Union

import torch
import torchaudio
from audiotools import AudioSignal

from dac.model.carver import Carver

SAMPLE_RATE = 24000
HOP_LENGTH = 960  # 24 kHz -> 25 Hz latent (encoder strides 3*5*8*8)


def _resolve_ckpt(ckpt_dir: Path):
    weights_p, meta_p = ckpt_dir / "weights.pth", ckpt_dir / "metadata.pth"
    if not (weights_p.exists() and meta_p.exists()):
        raise FileNotFoundError(
            f"expected weights.pth and metadata.pth in {ckpt_dir}"
        )
    return weights_p, meta_p


def load_model(ckpt_dir: Union[str, Path], device: str = "cuda") -> Carver:
    ckpt_dir = Path(ckpt_dir)
    weights_p, meta_p = _resolve_ckpt(ckpt_dir)

    meta = torch.load(str(meta_p), map_location="cpu", weights_only=False)
    kwargs = dict(meta.get("kwargs", {})) if isinstance(meta, dict) else {}
    assert kwargs, f"no kwargs inside metadata: {meta_p}"
    rb = kwargs.get("rate_bins")
    if isinstance(rb, torch.Tensor):
        kwargs["rate_bins"] = rb.tolist()

    model = Carver(**kwargs)
    state = torch.load(str(weights_p), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def load_audio(path: Union[str, Path], device: str = "cuda") -> torch.Tensor:
    """Load, downmix to mono, resample to 24 kHz, and normalize to -16 LUFS.

    Returns audio of shape ``[1, 1, L]`` on ``device``.
    """
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(wav)
    sig = AudioSignal(wav.unsqueeze(0), SAMPLE_RATE).normalize(-16).ensure_max_of_audio(1.0)
    audio = sig.audio_data.to(device)  # [1, 1, L]
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    return audio.float()


def energy_silence_ratio(ref: torch.Tensor, hop: int = HOP_LENGTH, db_thr: float = -40.0) -> float:
    """Fraction of frames whose RMS is below ``db_thr`` dBFS (silence estimate)."""
    ref = ref.reshape(-1)
    n = ref.shape[-1] // hop
    x = ref[: n * hop].reshape(n, hop)
    rms = torch.sqrt((x ** 2).mean(dim=1) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)
    return float((db < db_thr).float().mean())


@torch.no_grad()
def reconstruct(model: Carver, audio: torch.Tensor, rate: Union[str, float] = "auto"):
    """Reconstruct ``audio`` ([1,1,L]) at the requested rate.

    ``rate``:
      * ``"auto"``  -> per-utterance near-lossless budget (drops silence only).
      * ``1.0``     -> full frame rate (keep every frame).
      * ``0<r<1``   -> fixed keep-ratio; keeps the top ``round(r*T)`` frames.

    Returns ``(recon_waveform[L'], info_dict)``.
    """
    if isinstance(rate, str):
        assert rate == "auto", f"unknown rate '{rate}'"
        ref = audio.squeeze()
        sil = energy_silence_ratio(ref, HOP_LENGTH, -40.0)
        r = max(float(model.min_rate), 1.0 - sil)
    else:
        r = float(rate)
        assert 0.0 < r <= 1.0, f"rate must be 'auto' or a keep-ratio in (0, 1], got {r}"
    out = model(audio, manual_rate=r)
    info = {
        "requested_rate": r,
        "mask_rate": float(out.get("mask_rate", torch.tensor(1.0))),
    }
    return out["audio"].squeeze(), info


def save_audio(wav: torch.Tensor, path: Union[str, Path], sample_rate: int = SAMPLE_RATE):
    AudioSignal(wav.detach().cpu().reshape(1, 1, -1), sample_rate).write(str(path))
