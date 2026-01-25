import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .linear import Linear
from .train_bpe import train_bpe
from .tokenizer import Tokenizer
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU
from .rope import RotaryPositionalEmbedding

__all__ = [
    "Linear",
    "train_bpe",
    "Tokenizer",
    "Embedding",
    "RMSNorm",
    "SwiGLU",
    "RotaryPositionalEmbedding"
]