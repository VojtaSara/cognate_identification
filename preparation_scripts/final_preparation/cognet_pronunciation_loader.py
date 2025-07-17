import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Optional, Tuple, Dict, List
import time
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === CONFIG ===
DB_PATH = "cognates_2.db"
BATCH_SIZE = 1000
MAX_WORKERS = 4  # Number of parallel workers for JSON processing

def load_language_codes() -> Dict[str, str]:
    """Load language code mapping from JSON file."""
    try:
        with open('cognet_transfigure/language_codes.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading language codes: {e}")
        return {}

# Load language codes at module level
LANGUAGE_CODES = load_language_codes()

@dataclass
class WordData:
    """Data class to hold word information from JSON files."""
    language: str
    text: str
    lemma: str
    start_time: float
    end_time: float
    confidence: float
    source_file: str

def load_json_file(file_path: Path) -> Generator[WordData, None, None]:
    """Load and parse a single JSON file, yielding WordData objects."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        metadata = data['metadata']
        two_letter_code = metadata['language']
        # Convert to three letter code
        three_letter_code = LANGUAGE_CODES.get(two_letter_code, two_letter_code)
        source_file = metadata['source_file']
        
        # Create progress bar for this file
        words = data['words']
        pbar = tqdm(total=len(words), desc=f"Processing {file_path.name}", leave=False)
        
        for word_info in words:
            # Use lemma if it exists and is valid, otherwise use word
            lemma = word_info.get('lemma', '')
            if not lemma or lemma == '_':
                lemma = word_info['word'].strip()
            else:
                lemma = lemma.strip()
                
            yield WordData(
                language=three_letter_code,
                text=word_info['word'].strip(),
                lemma=lemma,
                start_time=word_info['start'],
                end_time=word_info['end'],
                confidence=word_info['confidence'],
                source_file=source_file
            )
            pbar.update(1)
        
        pbar.close()
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")

def process_json_files(json_dir: Path) -> Generator[WordData, None, None]:
    """Process all JSON files in a directory and its subdirectories."""
    json_files = list(json_dir.rglob('*.json'))
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    # Create progress bar for files
    with tqdm(total=len(json_files), desc="Processing files") as pbar:
        for file_path in json_files:
            yield from load_json_file(file_path)
            pbar.update(1)
            if pbar.n == 3:
                break

class PronunciationLoader:
    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        
        # Enable better performance
        self.cur.execute("PRAGMA journal_mode = WAL")
        self.cur.execute("PRAGMA synchronous = NORMAL")
        self.cur.execute("PRAGMA cache_size = -2000")
        
        # Create index for faster lookups if it doesn't exist
        self.cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_words_lang_text 
            ON words(language, text)
        """)
        self.conn.commit()
    
    def find_word_ids(self, language: str, text: str) -> List[int]:
        """Find all word_ids for a given language-text pair."""
        self.cur.execute("""
            SELECT word_id 
            FROM words 
            WHERE language = ? AND text = ?
        """, (language, text))
        return [row[0] for row in self.cur.fetchall()]
    
    def insert_pronunciations(self, word_data: Generator[WordData, None, None]):
        """Insert pronunciations for matching words."""
        start_time = time.time()
        batch = []
        total_inserted = 0
        total_processed = 0
        total_matched = 0
        
        for word in word_data:
            total_processed += 1
            
            # Try to find word_ids using lemma first, then the actual word
            word_ids = self.find_word_ids(word.language, word.lemma)
            if not word_ids:
                word_ids = self.find_word_ids(word.language, word.text)
            
            if word_ids:
                total_matched += 1
                # Add pronunciation for each matching word_id
                for word_id in word_ids:
                    batch.append((
                        word_id,
                        word.text,  # transcription
                        word.lemma,  # lemma
                        word.source_file,  # wave_path
                        None,  # chopped_word_path (not available in input)
                        word.start_time,
                        word.end_time
                    ))
                
                if len(batch) >= BATCH_SIZE:
                    self._insert_batch(batch)
                    total_inserted += len(batch)
                    batch = []
                    
                    if total_inserted % 10000 == 0:
                        elapsed = time.time() - start_time
                        logger.info(f"Processed {total_processed} words, matched {total_matched}, inserted {total_inserted} pronunciations ({total_inserted/elapsed:.2f} per second)")
        
        # Insert remaining batch
        if batch:
            self._insert_batch(batch)
            total_inserted += len(batch)
        
        elapsed = time.time() - start_time
        logger.info(f"\nFinal Statistics:")
        logger.info(f"Total words processed: {total_processed}")
        logger.info(f"Total words matched: {total_matched}")
        logger.info(f"Total pronunciations inserted: {total_inserted}")
        logger.info(f"Processing speed: {total_processed/elapsed:.2f} words/second")
        logger.info(f"Insertion speed: {total_inserted/elapsed:.2f} pronunciations/second")
    
    def _insert_batch(self, batch: list):
        """Insert a batch of pronunciations."""
        try:
            self.cur.executemany("""
                INSERT OR IGNORE INTO pronunciations 
                (word_id, transcription, lemma, wave_path, chopped_word_path, time_start, time_end)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, batch)
            self.conn.commit()
        except sqlite3.IntegrityError as e:
            logger.error(f"Error inserting batch: {e}")
            raise

def main():
    # Configuration
    json_dir = Path("../WhisperVault/transcriptions_lemmatized")  # Replace with actual path
    
    # Process JSON files and insert pronunciations
    logger.info("Starting pronunciation data processing...")
    loader = PronunciationLoader(DB_PATH)
    
    try:
        word_data = process_json_files(json_dir)
        loader.insert_pronunciations(word_data)
        logger.info("✅ Processing completed successfully!")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
    finally:
        loader.conn.close()

if __name__ == "__main__":
    main() 