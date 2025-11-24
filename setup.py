"""
Setup script for TensorTalk package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="tensortalk",
    version="1.0.0",
    author="Alexandre Dalban, Robert Richenburg, Arun Raja, Alexander Sharpe",
    author_email="dalban@bc.edu, richenbr@bc.edu, rajaar@bc.edu, sharpeal@bc.edu",
    description="A voice cloning framework for text-to-speech synthesis using WavLM features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robrich07/TensorTalk",
    project_urls={
        "Paper": "https://drive.google.com/file/d/1j9t6o2sKrWnu83dZkSFWDphGyeI9_NYr/view",
        "Bug Tracker": "https://github.com/robrich07/TensorTalk/issues",
    },
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "gTTS>=2.3.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
            "matplotlib>=3.7.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
        "eval": [
            "jiwer>=3.0.0",
            "scikit-learn>=1.3.0",
        ],
    },
    keywords=[
        "voice-cloning",
        "text-to-speech",
        "zero-shot",
        "speech-synthesis",
        "wavlm",
        "knn",
        "deep-learning",
        "tts",
    ],
)
