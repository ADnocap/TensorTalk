"""
SSL Encoder Module for TensorTalk

This module implements Self-Supervised Learning (SSL) feature extraction using WavLM-Large.
The encoder extracts 1024-dimensional features from the 6th transformer layer, which provides
an optimal balance between speaker identity and content information.
"""

import torch
import torchaudio
from transformers import WavLMModel
from typing import Optional


class SSLEncoder:
    """
    Self-Supervised Learning encoder using WavLM-Large for speech feature extraction.
    
    The encoder uses the 6th layer of WavLM-Large, which provides features that balance:
    - Speaker identity characteristics
    - Phonetic content information
    - Temporal context
    
    Attributes:
        device (str): Device to run the model on ('cuda' or 'cpu')
        model (WavLMModel): Pre-trained WavLM-Large model
        target_sample_rate (int): Target sample rate for audio (16000 Hz)
    """
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize the SSL encoder with WavLM-Large model.
        
        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_sample_rate = 16000
        
        print(f"Loading WavLM model to {self.device}...")
        self.model = WavLMModel.from_pretrained("microsoft/wavlm-large").to(self.device)
        self.model.eval()
        print("WavLM model loaded successfully!")
    
    @torch.no_grad()
    def extract_features(self, waveform: torch.Tensor, sample_rate: int = 16000) -> torch.Tensor:
        """
        Extract WavLM features from the 6th transformer layer.
        
        Args:
            waveform: Input audio waveform tensor of shape [channels, samples] or [samples]
            sample_rate: Sample rate of the input waveform
            
        Returns:
            Features tensor of shape [batch, time_steps, 1024]
            
        Note:
            - Automatically resamples to 16kHz if needed
            - Handles mono/stereo conversion
            - Each feature vector represents ~25ms of audio with 20ms stride
        """
        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.target_sample_rate
            )
        
        # Ensure proper batch dimension
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Move to device and extract features
        waveform = waveform.to(self.device)
        outputs = self.model(waveform, output_hidden_states=True)
        
        # Extract features from the 6th layer (optimal for voice cloning)
        features = outputs.hidden_states[6]
        
        return features
    
    def extract_from_file(self, audio_path: str) -> torch.Tensor:
        """
        Extract features directly from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Features tensor of shape [batch, time_steps, 1024]
        """
        waveform, sample_rate = torchaudio.load(audio_path)
        return self.extract_features(waveform, sample_rate)
