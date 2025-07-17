import json
from pathlib import Path
import logging
from typing import Dict, Set, Tuple
from tqdm import tqdm
import traceback

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_cognet(cognet_file: str) -> Dict[Tuple[str, str], str]:
    """Load COGNET data into a dictionary for fast lookups."""
    logger.info("Loading COGNET data...")
    cognet_dict = {}
    
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
                            cognet_dict[(word.lower(), lang.lower())] = concept_id
            except Exception as e:
                logger.warning(f"Skipping malformed line: {line.strip()} - Error: {str(e)}")
    
    logger.info(f"Loaded {len(cognet_dict)} word-language pairs from COGNET")
    return cognet_dict

def process_transcription(
    file_path: Path,
    cognet_dict: Dict[Tuple[str, str], str],
    output_dir: Path
) -> None:
    """Process a single transcription file and save to output directory."""
    try:
        logger.debug(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        language = data['metadata']['language']
        logger.debug(f"File language: {language}")
        
        modified = False
        
        for word in data['words']:
            try:
                # Try to use lemmatized word from JSON if present and valid
                original_word = word['word'].lower().strip()
                lemma_from_json = word.get('lemma', '').lower().strip()
                # Define what counts as invalid lemmatization
                invalid_lemmas = {'', '_', '—', '-', '–', '―', '…', '.', ',', ';', ':', '?', '!', '"', "'", '(', ')', '[', ']', '{', '}', '/'}
                use_lemma = lemma_from_json and lemma_from_json not in invalid_lemmas and lemma_from_json != original_word
                
                if use_lemma:
                    candidate_word = lemma_from_json
                    logger.debug(f"Using lemma from JSON: {candidate_word}")
                else:
                    candidate_word = original_word
                    logger.debug(f"Using original word: {candidate_word}")

                # Try to find concept ID using candidate word
                concept_id = cognet_dict.get((candidate_word, language.lower()))
                if not concept_id:
                    # Fallback: try original word
                    concept_id = cognet_dict.get((original_word, language.lower()))
                
                if concept_id:
                    word['concept_id'] = concept_id
                    word['lemma'] = candidate_word  # Store the lemma actually used
                    modified = True
                    logger.debug(f"Found concept ID {concept_id} for word {original_word}")
            except Exception as e:
                logger.error(f"Error processing word {word.get('word', 'UNKNOWN')}: {str(e)}\n{traceback.format_exc()}")
                continue
        
        if modified:
            # Create output directory if it doesn't exist
            output_path = output_dir / file_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to new file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved updated transcription to {output_path}")
            
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}\n{traceback.format_exc()}")

def main():
    try:
        # Configuration
        cognet_file = Path("CogNet-v2.0.tsv")
        transcriptions_dir = Path("transcriptions")
        output_dir = Path("transcriptions_with_concepts")
        
        logger.debug(f"COGNET file: {cognet_file}")
        logger.debug(f"Transcriptions directory: {transcriptions_dir}")
        logger.debug(f"Output directory: {output_dir}")
        
        # Load COGNET data
        cognet_dict = load_cognet(cognet_file)
        
        # Get list of transcription files
        transcription_files = [f for f in transcriptions_dir.glob('*.json') if f.name.endswith('_lemmatized.json')]
        logger.info(f"Found {len(transcription_files)} transcription files")
        
        # Process files
        for file_path in tqdm(transcription_files, desc="Processing transcriptions"):
            try:
                process_transcription(file_path, cognet_dict, output_dir)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}\n{traceback.format_exc()}")
                continue
                
    except Exception as e:
        logger.error(f"Fatal error in main: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    main() 