from faster_whisper import WhisperModel, BatchedInferencePipeline
import json
from datetime import datetime

model = WhisperModel("turbo", device="cuda", compute_type="float16")
batched_model = BatchedInferencePipeline(model=model)

print("works")
segments, info = batched_model.transcribe("HLAVNI SESTRIH.mp3", batch_size=16, word_timestamps=True, language="cs")

# Create a list to store all words with their timestamps
words_data = []
for segment in segments:
    if segment.words:
        for word in segment.words:
            words_data.append({
                "word": word.word,
                "start": round(word.start, 3),
                "end": round(word.end, 3),
                "confidence": round(word.probability, 3) if hasattr(word, 'probability') else None
            })

# Write to JSON file with timestamp in filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = f"transcription_{timestamp}.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "metadata": {
            "language": info.language,
            "language_probability": info.language_probability,
            "timestamp": timestamp
        },
        "words": words_data
    }, f, ensure_ascii=False, indent=2)

print(f"Transcription saved to {output_file}")
