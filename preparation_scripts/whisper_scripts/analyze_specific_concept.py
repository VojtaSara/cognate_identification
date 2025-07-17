import json
import os
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class WordInfo:
    word: str
    lemma: str
    start_time: float
    end_time: float
    source_file: str

@dataclass
class ConceptMatch:
    concept_id: str
    concept_name: str
    language: str
    words: List[WordInfo] = field(default_factory=list)

def load_cognet_data(cognet_path: str) -> Dict:
    """Load CogNet data from JSON file."""
    try:
        with open(cognet_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading CogNet data: {e}")
        return {}

def find_concept_matches(concept_id: str, cognet_data: Dict, lemmatized_dir: str) -> List[ConceptMatch]:
    """Find all matches for a specific concept across all languages."""
    matches = []
    
    # Get concept information
    concept_info = cognet_data.get(concept_id, {})
    if not concept_info:
        logger.error(f"Concept {concept_id} not found in CogNet data")
        return matches
    
    concept_name = concept_info.get('name', 'Unknown')
    logger.info(f"Searching for concept: {concept_name} ({concept_id})")
    
    # Get all lemmatized files
    lemmatized_files = list(Path(lemmatized_dir).glob('*_lemmatized.json'))
    
    for file_path in tqdm(lemmatized_files, desc="Processing files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract language from filename
            language = file_path.stem.split('_')[1]
            
            # Find words that match the concept
            concept_match = ConceptMatch(
                concept_id=concept_id,
                concept_name=concept_name,
                language=language,
                words=[]
            )
            
            for word_data in data.get('words', []):
                word = word_data.get('word', '')
                lemma = word_data.get('lemma', '')
                start_time = word_data.get('start', 0)
                end_time = word_data.get('end', 0)
                
                # Check if this word matches the concept
                if lemma in concept_info.get('lemmas', []):
                    word_info = WordInfo(
                        word=word,
                        lemma=lemma,
                        start_time=start_time,
                        end_time=end_time,
                        source_file=str(file_path)
                    )
                    concept_match.words.append(word_info)
            
            if concept_match.words:
                matches.append(concept_match)
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    
    return matches

def save_results(matches: List[ConceptMatch], output_file: str):
    """Save the results to a JSON file."""
    results = {
        'concept_id': matches[0].concept_id if matches else '',
        'concept_name': matches[0].concept_name if matches else '',
        'matches': [
            {
                'language': match.language,
                'words': [
                    {
                        'word': word.word,
                        'lemma': word.lemma,
                        'start_time': word.start_time,
                        'end_time': word.end_time,
                        'source_file': word.source_file
                    }
                    for word in match.words
                ]
            }
            for match in matches
        ]
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    # Configuration
    cognet_path = "cognet.json"
    lemmatized_dir = "lemmatized"
    output_file = "concept_matches.json"
    
    # Concept to search for (relief - n00354884)
    concept_id = "n00354884"
    
    # Load CogNet data
    cognet_data = load_cognet_data(cognet_path)
    if not cognet_data:
        return
    
    # Find matches
    matches = find_concept_matches(concept_id, cognet_data, lemmatized_dir)
    
    # Save results
    save_results(matches, output_file)
    
    # Print summary
    total_matches = sum(len(match.words) for match in matches)
    languages = len(matches)
    logger.info(f"Found {total_matches} matches across {languages} languages")

if __name__ == "__main__":
    main() 