"""
TensorTalk TTS Pipeline

This module implements the complete text-to-speech pipeline with zero-shot voice cloning.
It combines Google TTS for initial speech synthesis, WavLM for feature extraction,
k-NN matching for voice conversion, and HiFi-GAN for high-quality audio synthesis.
"""

import torch
import torchaudio
from gtts import gTTS
import tempfile
import os
from typing import List, Union, Optional
from pathlib import Path

from .ssl_encoder import SSLEncoder
from .knn_matcher import KNNMatcher


class TensorTalkPipeline:
    """
    Complete TTS pipeline with zero-shot voice cloning.
    
    This pipeline:
    1. Converts text to speech using Google TTS
    2. Extracts WavLM features from both source and target audio
    3. Performs voice conversion using k-NN matching
    4. Synthesizes final audio using HiFi-GAN vocoder
    
    Attributes:
        device (str): Device to run models on
        ssl_encoder (SSLEncoder): WavLM feature extractor
        knn_matcher (KNNMatcher): k-NN voice converter
        vocoder: HiFi-GAN vocoder for audio synthesis
    """
    
    def __init__(
        self, 
        k: int = 4,
        device: Optional[str] = None
    ):
        """
        Initialize the TensorTalk pipeline.
        
        Args:
            k: Number of nearest neighbors for voice conversion
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("Initializing TensorTalk pipeline...")
        
        # Initialize components
        self.ssl_encoder = SSLEncoder(device=self.device)
        self.knn_matcher = KNNMatcher(k=k, device=self.device)
        
        # Load HiFi-GAN vocoder from knn-vc
        print("Loading HiFi-GAN vocoder...")
        knn_vc = torch.hub.load(
            'bshall/knn-vc',
            'knn_vc',
            pretrained=True,
            prematched=True,
            trust_repo=True
        )
        self.vocoder = knn_vc.hifigan
        self.vocoder.to(self.device)
        
        print("Pipeline initialized successfully!")
    
    def text_to_speech(self, text: str, lang: str = 'en') -> torch.Tensor:
        """
        Convert text to speech using Google TTS.
        
        Args:
            text: Input text to synthesize
            lang: Language code (default: 'en')
            
        Returns:
            Audio waveform tensor
        """
        # Create temporary file for gTTS output
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Generate speech with gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(temp_path)
            
            # Load and return waveform
            waveform, sample_rate = torchaudio.load(temp_path)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            
            return waveform
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def load_target_features(
        self, 
        audio_paths: Union[str, List[str]], 
        trim_silence: bool = True
    ) -> torch.Tensor:
        """
        Load and extract features from target speaker audio files.
        
        Args:
            audio_paths: Path or list of paths to target speaker audio files
            trim_silence: Whether to trim silence from audio
            
        Returns:
            Concatenated target speaker features
        """
        if isinstance(audio_paths, str):
            audio_paths = [audio_paths]
        
        all_features = []
        
        for path in audio_paths:
            # Load audio
            waveform, sample_rate = torchaudio.load(path)
            
            # Trim silence if requested
            if trim_silence:
                transform = torchaudio.transforms.Vad(
                    sample_rate=sample_rate, 
                    trigger_level=7.0
                )
                # Trim from start
                waveform = transform(waveform)
                # Reverse, trim, and reverse back to trim from end
                waveform = torch.flip(waveform, (-1,))
                waveform = transform(waveform)
                waveform = torch.flip(waveform, (-1,))
            
            # Extract features
            features = self.ssl_encoder.extract_features(waveform, sample_rate)
            
            # Remove batch dimension if present
            if features.dim() == 3:
                features = features.squeeze(0)
            
            all_features.append(features)
        
        # Concatenate all features
        target_features = torch.cat(all_features, dim=0)
        
        return target_features
    
    @torch.no_grad()
    def synthesize(
        self, 
        text: str,
        target_audio_paths: Union[str, List[str]],
        alpha: float = 1.0,
        lang: str = 'en'
    ) -> torch.Tensor:
        """
        Synthesize speech in the target speaker's voice.
        
        Args:
            text: Text to synthesize
            target_audio_paths: Path(s) to target speaker audio files
            alpha: Voice conversion strength (0=source, 1=target)
            lang: Language code for gTTS
            
        Returns:
            Synthesized audio waveform
        """
        # Step 1: Generate initial speech with gTTS
        print("Generating initial speech with gTTS...")
        source_waveform = self.text_to_speech(text, lang=lang)
        
        # Step 2: Extract features from source speech
        print("Extracting source features...")
        source_features = self.ssl_encoder.extract_features(source_waveform, sample_rate=16000)
        if source_features.dim() == 3:
            source_features = source_features.squeeze(0)
        
        # Step 3: Load target speaker features
        print("Loading target speaker features...")
        target_features = self.load_target_features(target_audio_paths)
        
        # Step 4: Perform voice conversion
        print("Performing voice conversion...")
        converted_features = self.knn_matcher.convert_voice(
            source_features, 
            target_features, 
            alpha=alpha
        )
        
        # Step 5: Synthesize audio with vocoder
        print("Synthesizing audio...")
        # Add batch dimension for vocoder
        converted_features = converted_features.unsqueeze(0)
        
        # Generate audio
        with torch.no_grad():
            audio = self.vocoder(converted_features)
        
        # Remove batch and channel dimensions
        if audio.dim() > 1:
            audio = audio.squeeze()
        
        return audio
    
    def save_audio(self, waveform: torch.Tensor, path: str, sample_rate: int = 16000):
        """
        Save audio waveform to file.
        
        Args:
            waveform: Audio waveform tensor
            path: Output file path
            sample_rate: Sample rate for output audio
        """
        # Ensure waveform has correct shape [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Ensure waveform is on CPU
        waveform = waveform.cpu()
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        
        # Save audio
        torchaudio.save(path, waveform, sample_rate)
        print(f"Audio saved to {path}")
