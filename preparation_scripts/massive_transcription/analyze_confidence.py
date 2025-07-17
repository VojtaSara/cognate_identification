import json
import glob
import os

# Get the most recent transcription file
transcription_files = glob.glob("transcription_*.json")
latest_file = max(transcription_files, key=os.path.getctime)

# Read the JSON file
with open(latest_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Get all words with their confidence scores
words_with_confidence = [(word['word'], word['confidence'], word['start']) 
                        for word in data['words'] 
                        if word['confidence'] is not None]

# Sort by confidence (ascending) and get top 50
lowest_confidence_words = sorted(words_with_confidence, key=lambda x: x[1])[:100]

print(f"\nTop 50 words with lowest confidence from {latest_file}:")
print("-" * 80)
print(f"{'Word':<30} {'Confidence':<10} {'Timestamp':<10}")
print("-" * 80)
for word, confidence, timestamp in lowest_confidence_words:
    print(f"{word:<30} {confidence:<10.3f} {timestamp:<10.2f}s") 