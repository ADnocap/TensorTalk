"""
Simple example script for TensorTalk voice cloning.

Usage:
    python simple_example.py
"""

import sys
sys.path.append('.')

from src.pipeline import TensorTalkPipeline


def main():
    """Run a simple voice cloning example."""
    
    print("=" * 60)
    print("TensorTalk: Zero-Shot Voice Cloning")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = TensorTalkPipeline(k=4)
    
    # Define text to synthesize
    text = """Hello! This is a demonstration of TensorTalk, 
    a zero-shot voice cloning system developed at Boston College."""
    
    # Target speaker audio paths
    # Replace these with your own audio files
    target_audio_paths = [
        "examples/target_speaker/sample1.wav",
        # Add more files for better quality
    ]
    
    print(f"\n2. Text to synthesize:")
    print(f"   '{text}'\n")
    
    print("3. Target speaker files:")
    for path in target_audio_paths:
        print(f"   - {path}")
    
    # Synthesize speech
    print("\n4. Synthesizing speech...")
    try:
        audio = pipeline.synthesize(
            text=text,
            target_audio_paths=target_audio_paths,
            alpha=1.0  # Full voice conversion
        )
        
        # Save output
        output_path = "examples/generated/simple_output.wav"
        print(f"\n5. Saving audio to: {output_path}")
        pipeline.save_audio(audio, output_path)
        
        print("\n" + "=" * 60)
        print("SUCCESS! Audio generated successfully.")
        print("=" * 60)
        print(f"\nOutput saved to: {output_path}")
        print("You can play it with any audio player.")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: Could not find audio file.")
        print(f"   Please make sure the target audio files exist.")
        print(f"   Error details: {e}")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error during synthesis: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
