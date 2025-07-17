import json

def get_words_from_wordlevel(data):
    # Extract words from wordlevel format
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
    # Extract words from timestamped format
    return data['words']

# Load both transcriptions
with open('transcription_wordlevel.json', 'r', encoding='utf-8') as f:
    wordlevel = json.load(f)

with open('transcription_20250517_185828.json', 'r', encoding='utf-8') as f:
    timestamped = json.load(f)

# Get words from both formats
wordlevel_words = get_words_from_wordlevel(wordlevel)
timestamped_words = get_words_from_timestamped(timestamped)

# Compare lengths
print(f"Word count in wordlevel: {len(wordlevel_words)}")
print(f"Word count in timestamped: {len(timestamped_words)}")

# Compare word sequences
wordlevel_text = ' '.join(w['word'] for w in wordlevel_words)
timestamped_text = ' '.join(w['word'] for w in timestamped_words)

if wordlevel_text == timestamped_text:
    print("\nWord sequences are identical!")
else:
    print("\nWord sequences differ!")
    # Find first difference
    for i, (w1, w2) in enumerate(zip(wordlevel_words, timestamped_words)):
        if w1['word'] != w2['word']:
            print(f"\nFirst difference at word {i}:")
            print(f"Wordlevel:  {w1['word']} (start: {w1['start']:.2f}s)")
            print(f"Timestamped: {w2['word']} (start: {w2['start']:.2f}s)")
            break

# Compare confidence scores
wordlevel_conf = [w['confidence'] for w in wordlevel_words if w['confidence'] is not None]
timestamped_conf = [w['confidence'] for w in timestamped_words if w['confidence'] is not None]

if wordlevel_conf and timestamped_conf:
    print("\nConfidence score comparison:")
    print(f"Wordlevel average confidence: {sum(wordlevel_conf)/len(wordlevel_conf):.3f}")
    print(f"Timestamped average confidence: {sum(timestamped_conf)/len(timestamped_conf):.3f}") 