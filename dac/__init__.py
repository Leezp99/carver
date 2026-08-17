"""Minimal vendored `dac` package: the convolutional backbone Carver builds on.

Only the encoder blocks and NN layers required to run the Carver continuous
speech VAE are kept here. The package name ``dac`` is preserved so the absolute
imports inside the vendored files (``from dac.nn.layers import ...``) resolve.
"""

__version__ = "0.1.0"

from . import nn, model
from .model import Carver

__all__ = ["nn", "model", "Carver"]
