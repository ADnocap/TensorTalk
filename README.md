# TensorTalk: Zero-Shot Voice Cloning with WavLM and Rhythm Modeling

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://drive.google.com/file/d/1j9t6o2sKrWnu83dZkSFWDphGyeI9_NYr/view)

TensorTalk is a voice cloning framework that achieves high-quality zero-shot multi-speaker text-to-speech synthesis with reduced computational requirements. The system combines WavLM features, k-NN voice conversion, and optional rhythm modeling to clone voices from minimal reference audio.

## Key Features

- **Zero-Shot Voice Cloning**: Clone any voice from just seconds of reference audio
- **Efficient Architecture**: Uses Google TTS instead of computationally expensive GlowTTS
- **High Quality**: Achieves UTMOS score of 4.27 on LJSpeech dataset
- **Optional Rhythm Modeling**: Integrate Urhythmic framework for more natural speech patterns
- **Simple API**: Easy-to-use Python interface for quick experimentation

## Performance

Our system achieves competitive results while reducing training time:

| Metric  | TensorTalk (LJSpeech) | Previous SOTA |
| ------- | --------------------- | ------------- |
| UTMOS ↑ | **4.27**              | 4.16          |
| WER ↓   | **0.02**              | 2.76          |
| PER ↓   | **0.01**              | 0.78          |
| SECS ↑  | 0.65                  | 0.72          |

## Architecture

TensorTalk consists of four main components:

1. **Text-to-Speech (gTTS)**: Generates initial speech from text
2. **SSL Encoder (WavLM)**: Extracts 1024-dim features from layer 6
3. **k-NN Matcher**: Performs voice conversion via nearest neighbor matching
4. **Vocoder (HiFi-GAN)**: Synthesizes high-quality audio from features

## Technical Details

### WavLM Feature Extraction

- Uses Microsoft's WavLM-Large model (94k hours of training data)
- Extracts features from the 6th transformer layer
- Each feature vector represents 25ms of audio with 20ms stride
- Output dimension: 1024

### k-NN Voice Conversion

- Finds k=4 nearest neighbors using cosine similarity
- Matches phonetically similar frames between speakers
- Preserves content while transferring voice characteristics
- Optional linear interpolation for voice mixing

### Vocoder

- HiFi-GAN vocoder from knn-vc implementation
- Trained to reconstruct audio from WavLM features
- Generates high-quality 16kHz audio output

## Citation

If you use TensorTalk in your research, please cite our paper:

Full paper available at: [Google Drive](https://drive.google.com/file/d/1j9t6o2sKrWnu83dZkSFWDphGyeI9_NYr/view)

## Related Work

This project builds upon several key works:

- **SSL-TTS** (Hajal et al., 2024): Framework for zero-shot TTS using self-supervised learning
- **kNN-VC** (Baas et al., 2023): Simple yet effective voice conversion with k-NN
- **WavLM** (Chen et al., 2022): Large-scale self-supervised speech model
- **Urhythmic** (van Niekerk et al., 2023): Rhythm modeling for voice conversion

## Datasets

The system was evaluated on:

- **LJSpeech**: Single speaker, 13,100 sentences
- **VCTK**: 110 speakers, 400 sentences each

## Contributing

This is a research project performed at Boston College. We welcome feedback!

## Authors

- Alexandre Dalban (dalban@bc.edu)
- Robert Richenburg (richenbr@bc.edu)
- Arun Raja (rajaar@bc.edu)
- Alexander Sharpe (sharpeal@bc.edu)

Boston College, 2024
