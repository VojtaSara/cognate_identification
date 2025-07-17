import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, Set, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cognet(cognet_file: str) -> Tuple[Dict[str, Set[Tuple[str, str]]], Dict[str, str]]:
    """Load COGNET data into dictionaries for fast lookups."""
    logger.info("Loading COGNET data...")
    concept_to_words = defaultdict(set)  # concept_id -> set of (word, language) pairs
    word_to_concept = {}  # (word, language) -> concept_id
    
    with open(cognet_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                fields = line.strip().split('\t')
                if len(fields) < 3:
                    continue
                    
                concept_id = fields[0]
                
                # Process each language-word pair
                for i in range(1, len(fields), 2):
                    if i + 1 < len(fields):
                        lang = fields[i]
                        word = fields[i + 1]
                        if word:  # Only add if word is not empty
                            word = word.lower()
                            lang = lang.lower()
                            concept_to_words[concept_id].add((word, lang))
                            word_to_concept[(word, lang)] = concept_id
            except Exception as e:
                logger.warning(f"Skipping malformed line: {line.strip()} - Error: {str(e)}")
    
    logger.info(f"Loaded COGNET data with {len(concept_to_words)} concepts")
    return concept_to_words, word_to_concept

def count_concepts(directory: Path) -> Dict[str, int]:
    """Count concept occurrences across all transcription files."""
    concept_counter = Counter()
    total_files = 0
    files_with_concepts = 0
    
    # Process all JSON files in the directory
    for file_path in directory.glob('*.json'):
        total_files += 1
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count concepts in this file
            file_concepts = set()
            for word in data['words']:
                if 'concept_id' in word:
                    concept_counter[word['concept_id']] += 1
                    file_concepts.add(word['concept_id'])
            
            if file_concepts:
                files_with_concepts += 1
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    logger.info(f"Processed {total_files} files, {files_with_concepts} contained concepts")
    return concept_counter

def main():
    # Configuration
    cognet_file = "PodcastRealDataset/wordleveltranscripts/CogNet-v2.0.tsv"
    transcriptions_dir = Path("WhisperVault/transcriptions_with_concepts")
    
    # Load COGNET data
    concept_to_words, _ = load_cognet(cognet_file)
    
    # Count concepts
    concept_counts = count_concepts(transcriptions_dir)
    
    # Print top 20 most common concepts with their associated words
    print("\nTop 20 most common concepts:")
    print("-" * 80)
    for concept_id, count in concept_counts.most_common(20):
        print(f"\nConcept ID: {concept_id:15} | Count: {count:5}")
        print("Associated words:")
        words_by_lang = defaultdict(list)
        for word, lang in concept_to_words[concept_id]:
            words_by_lang[lang].append(word)
        
        for lang, words in sorted(words_by_lang.items()):
            print(f"  {lang:5}: {', '.join(sorted(words))}")

if __name__ == "__main__":
    main()
