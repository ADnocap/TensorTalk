"""
TensorTalk: A Voice Cloning Framework for Text-to-Speech Synthesis

TensorTalk is a zero-shot voice cloning system that combines:
- WavLM features for robust speech representation
- k-NN matching for efficient voice conversion
- Google TTS for initial speech synthesis
- HiFi-GAN vocoder for high-quality audio generation

For more details, see the paper at:
https://drive.google.com/file/d/1j9t6o2sKrWnu83dZkSFWDphGyeI9_NYr/view
"""

from .ssl_encoder import SSLEncoder
from .knn_matcher import KNNMatcher
from .pipeline import TensorTalkPipeline

__version__ = "1.0.0"
__author__ = "Alexandre Dalban, Robert Richenburg, Arun Raja, Alexander Sharpe"

__all__ = [
    'SSLEncoder',
    'KNNMatcher',
    'TensorTalkPipeline',
]
