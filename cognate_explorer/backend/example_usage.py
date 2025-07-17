#!/usr/bin/env python3
"""
Example usage of the similarity analysis functionality.
This script demonstrates how to use the SimilarityAnalyzer class.
"""

import json
from pathlib import Path
import sys

# Add the current directory to the path
sys.path.append(str(Path(__file__).parent))

from similarity_analyzer import SimilarityAnalyzer

def create_sample_transcript():
    """Create a sample transcript for testing."""
    sample_transcript = {
        "metadata": {
            "language": "en",
            "language_probability": 0.95,
            "timestamp": "20250101_120000"
        },
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
            {"word": "world", "start": 0.6, "end": 1.1, "confidence": 0.8},
            {"word": "how", "start": 1.2, "end": 1.7, "confidence": 0.85},
            {"word": "are", "start": 1.8, "end": 2.3, "confidence": 0.75},
            {"word": "you", "start": 2.4, "end": 2.9, "confidence": 0.9}
        ]
    }
    
    # Save sample transcript
    transcript_path = Path("temp_transcriptions/sample_transcript.json")
    transcript_path.parent.mkdir(exist_ok=True)
    
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(sample_transcript, f, ensure_ascii=False, indent=2)
    
    return str(transcript_path)

def example_similarity_analysis():
    """Example of how to use the similarity analyzer."""
    
    # Create sample transcript
    transcript_path = create_sample_transcript()
    print(f"📝 Created sample transcript: {transcript_path}")
    
    # You would need an actual audio file for this to work
    audio_path = "uploads/sample_audio.wav"  # This would need to exist
    
    # Model paths (these should exist in your project)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    model_path = str(project_root / "experiments/exp_01_baseline/checkpoints/model_best_epoch_23_20250703_153314.pt")
    config_path = str(project_root / "experiments/exp_01_baseline/config.yaml")
    
    print(f"🎯 Model path: {model_path}")
    print(f"⚙️ Config path: {config_path}")
    
    # Check if files exist
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return
    
    if not Path(audio_path).exists():
        print(f"❌ Audio file not found: {audio_path}")
        print("💡 You need to provide an actual audio file for testing")
        return
    
    try:
        # Initialize the analyzer
        print("🔧 Initializing SimilarityAnalyzer...")
        analyzer = SimilarityAnalyzer(model_path, config_path)
        
        # Analyze similarity
        print("🔍 Analyzing transcript similarity...")
        result = analyzer.analyze_transcript_similarity(transcript_path, audio_path)
        
        # Display results
        print("\n📊 Similarity Analysis Results:")
        print("=" * 50)
        print(f"Reference word: '{result['reference_word']}'")
        print(f"Reference mel path: {result['reference_mel_path']}")
        print("\nRanked similarities:")
        
        for i, item in enumerate(result['similarities'], 1):
            print(f"{i}. '{item['word']}' (start: {item['start']:.2f}s)")
            print(f"   Similarity: {item['similarity_score']:.4f}")
            print(f"   Distance: {item['distance']:.4f}")
            print(f"   Mel path: {item['mel_path']}")
            print()
        
        print(f"📄 Updated transcript saved to: {result['updated_transcript_path']}")
        
    except Exception as e:
        print(f"❌ Error during similarity analysis: {e}")
        import traceback
        traceback.print_exc()

def show_api_response_example():
    """Show an example of what the API response looks like."""
    
    example_response = {
        "status": "done",
        "result": {
            "whisper": {
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
                    {"word": "world", "start": 0.6, "end": 1.1, "confidence": 0.8}
                ],
                "metadata": {
                    "language": "en",
                    "language_probability": 0.95,
                    "timestamp": "20250101_120000"
                }
            },
            "whisper_json_path": "temp_transcriptions/transcription_20250101_120000.json",
            "mel_json_path": "temp_transcriptions/transcription_20250101_120000_with_mel.json",
            "similarity_analysis": {
                "reference_word": "hello",
                "reference_mel_path": "mel_spectrograms/word_0.pt",
                "similarities": [
                    {
                        "word": "world",
                        "start": 0.6,
                        "end": 1.1,
                        "similarity_score": 0.85,
                        "distance": 0.15,
                        "mel_path": "mel_spectrograms/word_1.pt"
                    }
                ],
                "updated_transcript_path": "temp_transcriptions/transcription_20250101_120000_with_similarity.json"
            }
        }
    }
    
    print("📋 Example API Response:")
    print("=" * 50)
    print(json.dumps(example_response, indent=2))

if __name__ == "__main__":
    print("🚀 Similarity Analysis Example")
    print("=" * 50)
    
    # Show API response example
    show_api_response_example()
    print("\n" + "=" * 50)
    
    # Run example analysis (if files exist)
    example_similarity_analysis() 