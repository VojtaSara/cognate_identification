import json
import csv
from pathlib import Path

csv.field_size_limit(27483647)  # Use maximum value for 32-bit systems

def load_cognet(cognet_path, target_lang='sv'):
    """
    Load CogNet database into a set for O(1) lookups
    Only includes words from the target language
    """
    cognet_words = set()
    with open(cognet_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:  # Ensure row has enough columns
                lang_code = row[1].lower()
                word = row[2].lower()
                if lang_code == target_lang and word:
                    cognet_words.add(word)
    return cognet_words

def analyze_transcription(json_path, cognet_words):
    """Analyze a transcription JSON file against CogNet"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    lemmata = []

    print(data)
    
    # Extract words and lemmata
    for item in data['words']:
        # Clean the word (remove leading/trailing spaces and punctuation)
        word = item['word'].strip().strip('.,!?').lower()
        lemma = item['lemma'].strip().strip('.,!?').lower()
        
        if word:  # Skip empty words
            words.append(word)
        if lemma:  # Skip empty lemmata
            lemmata.append(lemma)
    
    # Count matches
    word_matches = sum(1 for word in words if word in cognet_words)
    lemma_matches = sum(1 for lemma in lemmata if lemma in cognet_words)
    
    # Get lists of matched and unmatched words/lemmata
    matched_words = [word for word in words if word in cognet_words]
    unmatched_words = [word for word in words if word not in cognet_words]
    matched_lemmata = [lemma for lemma in lemmata if lemma in cognet_words]
    unmatched_lemmata = [lemma for lemma in lemmata if lemma not in cognet_words]
    
    return {
        'total_words': len(words),
        'word_matches': word_matches,
        'word_match_percentage': (word_matches / len(words) * 100) if words else 0,
        'total_lemmata': len(lemmata),
        'lemma_matches': lemma_matches,
        'lemma_match_percentage': (lemma_matches / len(lemmata) * 100) if lemmata else 0,
        'matched_words': matched_words,
        'unmatched_words': unmatched_words,
        'matched_lemmata': matched_lemmata,
        'unmatched_lemmata': unmatched_lemmata
    }

def main():
    # Paths
    cognet_path = Path("./CogNet-v2.0.tsv")
    #json_path = Path("WhisperVault/transcriptions/65729845_en_7309884_feed_lemmatized.json")
    json_path = Path("WhisperVault/transcriptions/a5e4541f-6a1a-448d-81e2-e5701486da92_cs_6747336_rss_lemmatized.json")

    # Load CogNet
    print("Loading CogNet database...")
    cognet_words = load_cognet(cognet_path, target_lang='ces')
    print(f"Loaded {len(cognet_words)} words from CogNet")
    
    # Analyze transcription
    print("\nAnalyzing transcription...")
    results = analyze_transcription(json_path, cognet_words)
    
    # Print results
    print("\nResults:")
    print(f"Total words analyzed: {results['total_words']}")
    print(f"Words found in CogNet: {results['word_matches']} ({results['word_match_percentage']:.2f}%)")
    print(f"\nTotal lemmata analyzed: {results['total_lemmata']}")
    print(f"Lemmata found in CogNet: {results['lemma_matches']} ({results['lemma_match_percentage']:.2f}%)")
    
    # Print some examples
    print("\nExample matched words:", results['matched_words'][:10] if results['matched_words'] else "None")
    print("Example unmatched words:", results['unmatched_words'][:10] if results['unmatched_words'] else "None")
    print("\nExample matched lemmata:", results['matched_lemmata'][:10] if results['matched_lemmata'] else "None")
    print("Example unmatched lemmata:", results['unmatched_lemmata'][:10] if results['unmatched_lemmata'] else "None")

if __name__ == "__main__":
    main() 