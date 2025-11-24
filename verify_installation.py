"""
Installation verification script for TensorTalk.

This script checks that all required dependencies are installed and working.
Run this after installing the package to verify everything is set up correctly.

Usage:
    python verify_installation.py
"""

import sys


def check_imports():
    """Check if all required packages can be imported."""
    print("Checking package imports...")
    print("-" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'torchaudio': 'TorchAudio',
        'transformers': 'Transformers',
        'gtts': 'gTTS',
        'numpy': 'NumPy',
    }
    
    failed = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name:20s} ... OK")
        except ImportError as e:
            print(f"✗ {name:20s} ... FAILED")
            failed.append((name, package))
    
    return failed


def check_cuda():
    """Check CUDA availability."""
    print("\nChecking CUDA...")
    print("-" * 60)
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
        else:
            print("⚠ CUDA not available (CPU only)")
            print("  This is okay, but processing will be slower.")
        
        return True
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False


def check_models():
    """Check if models can be loaded."""
    print("\nChecking model loading...")
    print("-" * 60)
    
    try:
        from transformers import WavLMModel
        print("✓ WavLM model ... Loading (this may take a moment)")
        
        # Try to load WavLM (this will download if not cached)
        try:
            model = WavLMModel.from_pretrained("microsoft/wavlm-large")
            print("✓ WavLM model ... Loaded successfully")
            del model  # Free memory
            return True
        except Exception as e:
            print(f"✗ WavLM model ... Failed to load: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def check_tensortalk():
    """Check if TensorTalk package can be imported."""
    print("\nChecking TensorTalk package...")
    print("-" * 60)
    
    try:
        import sys
        sys.path.append('.')
        from src import TensorTalkPipeline, SSLEncoder, KNNMatcher
        
        print("✓ TensorTalkPipeline ... OK")
        print("✓ SSLEncoder ... OK")
        print("✓ KNNMatcher ... OK")
        
        return True
    except ImportError as e:
        print(f"✗ Failed to import TensorTalk: {e}")
        print("\nMake sure you're running this from the TensorTalk root directory.")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("TensorTalk Installation Verification")
    print("=" * 60)
    print()
    
    results = []
    
    # Check imports
    failed_imports = check_imports()
    results.append(("Package Imports", len(failed_imports) == 0))
    
    # Check CUDA
    cuda_ok = check_cuda()
    results.append(("CUDA Check", cuda_ok))
    
    # Check TensorTalk
    tensortalk_ok = check_tensortalk()
    results.append(("TensorTalk Import", tensortalk_ok))
    
    # Check model loading (optional, can be slow)
    print("\n" + "=" * 60)
    response = input("Load WavLM model to test? (y/n, this may take time): ").lower()
    if response == 'y':
        model_ok = check_models()
        results.append(("Model Loading", model_ok))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_ok = True
    for name, status in results:
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
        if not status:
            all_ok = False
    
    print("=" * 60)
    
    if failed_imports:
        print("\n⚠ Missing packages:")
        for name, package in failed_imports:
            print(f"   - {name} (pip install {package})")
        print("\nInstall missing packages with:")
        print("   pip install -r requirements.txt")
    
    if all_ok:
        print("\n✓ All checks passed! TensorTalk is ready to use.")
        print("\nNext steps:")
        print("  1. Check out notebooks/demo.ipynb for a tutorial")
        print("  2. Run simple_example.py for a quick test")
        print("  3. Read the paper: TensorTalk_Paper.pdf")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
