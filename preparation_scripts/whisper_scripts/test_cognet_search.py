"""
Test script for searching words in CogNet database.
This script demonstrates how to efficiently search for words from a lemmatized JSON file
in the CogNet database for a specific language.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Increase CSV field size limit for large files
csv.field_size_limit(27483647)

@dataclass(frozen=True)  # Make the class immutable and hashable
class WordInfo:
    """Class to store information about a word from the transcription."""
    word: str
    lemma: str
    start: float
    end: float
    source_file: str

@dataclass
class ConceptMatch:
    """Class to store information about a concept match from CogNet."""
    concept_id: str
    language: str
    words: List[WordInfo]

class CogNetSearcher:
    """Class to handle CogNet database operations and word searching."""
    
    def __init__(self, cognet_path: Path):
        """Initialize the searcher with CogNet database path."""
        self.cognet_path = cognet_path
        self.word_to_concepts: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self._load_cognet()
    
    def _load_cognet(self) -> None:
        """Load CogNet database into memory for efficient searching."""
        print(f"Loading CogNet database from {self.cognet_path}...")
        
        with open(self.cognet_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                if len(row) >= 4:
                    concept_id = row[0]
                    lang_code = row[1].lower()
                    word = row[2].lower().strip()
                    
                    if word:  # Skip empty words
                        self.word_to_concepts[word].append((concept_id, lang_code))
        
        print(f"Loaded {len(self.word_to_concepts)} unique words from CogNet")
    
    def find_matches(self, words: List[WordInfo], target_lang: str) -> List[ConceptMatch]:
        """
        Find matches for the given words in CogNet for the target language.
        
        Args:
            words: List of WordInfo objects to search for
            target_lang: Target language code to match against
            
        Returns:
            List of ConceptMatch objects containing the matches
        """
        # Dictionary to store matches by concept
        concept_matches: Dict[str, Dict[str, Set[WordInfo]]] = defaultdict(
            lambda: defaultdict(set)
        )
        
        # Process each word
        for word_info in words:
            # Check both word and lemma
            for test_word in [word_info.word, word_info.lemma]:
                test_word = test_word.strip().strip('.,!?').lower()
                
                if test_word in self.word_to_concepts:
                    for concept_id, lang_code in self.word_to_concepts[test_word]:
                        if lang_code == target_lang:
                            concept_matches[concept_id][lang_code].add(word_info)
        
        # Convert to ConceptMatch objects
        return [
            ConceptMatch(
                concept_id=concept_id,
                language=next(iter(lang_matches.keys())),  # Get the language code
                words=list(next(iter(lang_matches.values())))  # Get the words
            )
            for concept_id, lang_matches in concept_matches.items()
        ]

def load_transcription(json_path: Path) -> List[WordInfo]:
    """Load and process a transcription JSON file."""
    print(f"Loading transcription from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    for item in data['words']:
        word_info = WordInfo(
            word=item['word'].strip(),
            lemma=item['lemma'].strip(),
            start=item['start'],
            end=item['end'],
            source_file=json_path.name
        )
        words.append(word_info)
    
    print(f"Loaded {len(words)} words from transcription")
    return words

def main():
    """Main function to demonstrate the word search functionality."""
    # Configuration
    cognet_path = Path("./CogNet-v2.0.tsv")
    json_path = Path("WhisperVault/transcriptions/65729845_en_7309884_feed_lemmatized.json")
    target_language = "eng"  # Target language code
    
    # Initialize searcher
    searcher = CogNetSearcher(cognet_path)
    
    # Load transcription
    words = load_transcription(json_path)
    
    # Find matches
    print(f"\nSearching for matches in {target_language}...")
    matches = searcher.find_matches(words, target_language)
    
    # Print results
    print(f"\nFound {len(matches)} concept matches:")
    for match in matches:
        print(f"\nConcept {match.concept_id} ({match.language}):")
        for word in match.words:
            print(f"  - {word.word} (lemma: {word.lemma}) "
                  f"[{word.start:.2f}-{word.end:.2f}] "
                  f"from {word.source_file}")

if __name__ == "__main__":
    main() 