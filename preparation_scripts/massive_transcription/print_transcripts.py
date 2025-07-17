import json

def get_words_from_wordlevel(data):
    words = []
    for segment in data['segments']:
        if 'words' in segment:
            for word in segment['words']:
                words.append({
                    'word': word['word'],
                    'start': word['start'],
                    'end': word['end'],
                    'confidence': word.get('probability', None)
                })
    return words

def get_words_from_timestamped(data):
    return data['words']

# Load both transcriptions
with open('transcription_wordlevel.json', 'r', encoding='utf-8') as f:
    wordlevel = json.load(f)

with open('transcription_20250517_185828.json', 'r', encoding='utf-8') as f:
    timestamped = json.load(f)

# Get words from both formats
wordlevel_words = get_words_from_wordlevel(wordlevel)
timestamped_words = get_words_from_timestamped(timestamped)

print("\n=== Wordlevel Transcription ===")
print("Time\t\tWord\t\tConfidence")
print("-" * 50)
for word in wordlevel_words:
    conf = f"{word['confidence']:.3f}" if word['confidence'] is not None else "N/A"
    print(f"{word['start']:.2f}s\t{word['word']:<15}\t{conf}")

print("\n\n=== Timestamped Transcription ===")
print("Time\t\tWord\t\tConfidence")
print("-" * 50)
for word in timestamped_words:
    conf = f"{word['confidence']:.3f}" if word['confidence'] is not None else "N/A"
    print(f"{word['start']:.2f}s\t{word['word']:<15}\t{conf}") 