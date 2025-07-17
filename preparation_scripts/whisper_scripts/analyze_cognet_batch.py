import json
import csv
from pathlib import Path
from collections import defaultdict
import glob

csv.field_size_limit(27483647)  # Use maximum value for 32-bit systems

def load_cognet(cognet_path):
    """
    Load CogNet database into a dictionary mapping words to their concepts
    Returns a dict where key is word and value is list of (concept_id, lang_code) tuples
    """
    cognet_data = defaultdict(list)
    with open(cognet_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 4:  # Ensure row has enough columns
                concept_id = row[0]
                lang_code = row[1].lower()
                word = row[2].lower()
                if word:  # Skip empty words
                    cognet_data[word].append((concept_id, lang_code))
    return cognet_data

def process_transcription(json_path, cognet_data):
    """Process a single transcription JSON file and return matches with CogNet"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Dictionary to store matches by concept
    concept_matches = defaultdict(lambda: defaultdict(set))  # Using set to prevent duplicates
    
    # Get source file info
    source_file = Path(json_path).name
    
    # Process each word
    for item in data['words']:
        # Clean the word and lemma
        word = item['word'].strip().strip('.,!?').lower()
        lemma = item['lemma'].strip().strip('.,!?').lower()
        
        # Skip empty words
        if not word:
            continue
            
        # Create word info dictionary
        word_info = {
            'word': word,
            'lemma': lemma,
            'start': item['start'],
            'end': item['end'],
            'source_file': source_file
        }
        
        # Convert to tuple for set storage
        word_info_tuple = tuple(sorted(word_info.items()))
        
        # Check both word and lemma against CogNet, but only add each word once per concept
        matched_concepts = set()
        for test_word in [word, lemma]:
            if test_word in cognet_data:
                for concept_id, lang_code in cognet_data[test_word]:
                    if concept_id not in matched_concepts:
                        concept_matches[concept_id][lang_code].add(word_info_tuple)
                        matched_concepts.add(concept_id)
    
    # Convert sets back to lists of dictionaries
    return {
        concept_id: {
            lang_code: [dict(word_info) for word_info in word_infos]
            for lang_code, word_infos in lang_matches.items()
        }
        for concept_id, lang_matches in concept_matches.items()
    }

def process_directory(json_dir, cognet_path, output_path):
    """Process all JSON files in a directory and save results"""
    # Load CogNet database
    print("Loading CogNet database...")
    cognet_data = load_cognet(cognet_path)
    print(f"Loaded CogNet database with {len(cognet_data)} unique words")
    
    # Get all JSON files
    json_files = glob.glob(str(Path(json_dir) / "**/*_lemmatized.json"), recursive=True)
    print(f"Found {len(json_files)} JSON files to process")
    
    # Process each file
    all_concept_matches = defaultdict(lambda: defaultdict(list))
    
    for json_file in json_files:
        print(f"Processing {json_file}...")
        file_matches = process_transcription(json_file, cognet_data)
        
        # Merge matches into main dictionary
        for concept_id, lang_matches in file_matches.items():
            for lang_code, words in lang_matches.items():
                all_concept_matches[concept_id][lang_code].extend(words)
    
    # Convert to final format
    final_output = {}
    for concept_id, lang_matches in all_concept_matches.items():
        final_output[concept_id] = {
            lang_code: words for lang_code, words in lang_matches.items()
        }
    
    # Save results
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")

def main():
    # Paths
    cognet_path = Path("./CogNet-v2.0.tsv")
    json_dir = Path("WhisperVault/transcriptions")
    output_path = Path("WhisperVault/cognet_matches.json")
    
    # Process files
    process_directory(json_dir, cognet_path, output_path)

if __name__ == "__main__":
    main() 