import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import json
import os
import gc
import sys

SAMPLE_RATE = 16000
PADDING_MS = 83
MEL_DIR = Path("mel_spectrograms")
MEL_DIR.mkdir(exist_ok=True)

def cut_mel_segments(audio_path, words, sample_rate=SAMPLE_RATE, padding_ms=PADDING_MS):
    """For each word, cut the audio segment and save its MEL spectrogram."""
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return []

    mel_paths = []
    for idx, word in enumerate(words):
        start = max(0, word["start"] - padding_ms / 1000.0)
        duration = (word["end"] - word["start"]) + 2 * padding_ms / 1000.0
        frame_offset = int(start * sample_rate)
        num_frames = int(duration * sample_rate)
        if num_frames > sample_rate * 10:
            num_frames = sample_rate * 10
        segment = waveform[frame_offset:frame_offset+num_frames]
        if len(segment) < 100:
            mel_paths.append(None)
            continue
        mel = MelSpectrogram(sample_rate=sample_rate, n_mels=80)(segment)
        mel_db = AmplitudeToDB()(mel)
        mel_path = MEL_DIR / f"word_{idx}.pt"
        torch.save(mel_db.clone(), mel_path)
        mel_paths.append(str(mel_path))
        del mel, mel_db, segment
        gc.collect()
    return mel_paths

def add_mel_paths_to_transcript(json_path, audio_path):
    """
    Given a transcript JSON and audio file, cut MELs for each word and add their paths to the JSON.
    Returns the path to the updated JSON file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    words = data["words"] if "words" in data else data.get("result", {}).get("words", [])
    mel_paths = cut_mel_segments(audio_path, words)
    for word, mel_path in zip(words, mel_paths):
        word["mel_path"] = mel_path
    # Save updated JSON
    out_json = Path(json_path).with_name(Path(json_path).stem + "_with_mel.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated transcript with MEL paths saved to {out_json}")
    return str(out_json)

# CLI usage preserved
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python mel_from_transcript.py <transcript.json> <audiofile>")
        sys.exit(1)
    add_mel_paths_to_transcript(sys.argv[1], sys.argv[2]) 