# Similarity Analysis Integration

This document describes the new similarity analysis functionality that has been integrated into the backend.

## Overview

The system now performs the following steps after creating a transcription:

1. **Transcription**: Uses Whisper to create word-level transcriptions with timestamps
2. **Mel Spectrogram Cutting**: Cuts individual mel spectrograms for each word in the transcript
3. **Similarity Analysis**: Uses a trained PyTorch model to compute similarity scores between words
4. **Ranking**: Ranks all words by their similarity to the first word in the transcript

## Components

### 1. SimilarityAnalyzer Class (`similarity_analyzer.py`)

The main class that handles:
- Loading the trained PyTorch model
- Cutting mel spectrograms from audio segments
- Computing similarity scores between spectrograms
- Ranking words by similarity

### 2. Integration with Main Backend (`main.py`)

The similarity analysis is automatically triggered after transcription in the `process_job` function.

## How It Works

### Step 1: Mel Spectrogram Cutting
- For each word in the transcript, extracts the audio segment based on start/end timestamps
- Applies padding (83ms) around each word
- Converts to mel spectrogram with 80 mel bins
- Saves as `.pt` files in the `mel_spectrograms/` directory

### Step 2: Similarity Computation
- Uses the first word as a reference
- Compares the reference word's spectrogram with all other words
- Uses a trained Transformer-based Siamese network
- Computes both cosine similarity and Euclidean distance

### Step 3: Ranking
- Sorts all words by similarity score (highest first)
- Returns detailed results including:
  - Reference word information
  - Similarity scores for all other words
  - Mel spectrogram file paths
  - Updated transcript with mel paths

## API Response

The API now returns additional information in the job result:

```json
{
  "status": "done",
  "result": {
    "whisper": { /* whisper transcription data */ },
    "whisper_json_path": "path/to/transcript.json",
    "mel_json_path": "path/to/transcript_with_mel.json",
    "similarity_analysis": {
      "reference_word": "first",
      "reference_mel_path": "mel_spectrograms/word_0.pt",
      "similarities": [
        {
          "word": "second",
          "start": 1.2,
          "end": 1.8,
          "similarity_score": 0.85,
          "distance": 0.15,
          "mel_path": "mel_spectrograms/word_1.pt"
        },
        // ... more words ranked by similarity
      ],
      "updated_transcript_path": "path/to/transcript_with_similarity.json"
    }
  }
}
```

## Model Configuration

The system uses:
- **Model**: `experiments/exp_01_baseline/checkpoints/model_best_epoch_23_20250703_153314.pt`
- **Config**: `experiments/exp_01_baseline/config.yaml`
- **Architecture**: Transformer-based Siamese network
- **Input**: 80 mel spectrograms
- **Output**: 128-dimensional embeddings

## File Structure

```
slovoshop/backend/
├── main.py                    # Main FastAPI application
├── similarity_analyzer.py     # Similarity analysis module
├── mel_from_transcript.py     # Original mel cutting module
├── mel_spectrograms/         # Generated mel spectrograms
├── temp_transcriptions/       # Transcript files with mel paths
└── uploads/                  # Uploaded audio files
```

## Testing

Run the test script to verify functionality:

```bash
cd slovoshop/backend
python test_similarity.py
```

## Dependencies

The system requires:
- PyTorch and torchaudio (already in requirements.txt)
- The trained model checkpoint file
- The model configuration file

## Error Handling

The system includes comprehensive error handling:
- Graceful fallback if mel cutting fails
- Error reporting for individual word processing
- Detailed error messages in API responses
- Memory management with garbage collection 