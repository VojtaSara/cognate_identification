import os
from pathlib import Path
from faster_whisper import WhisperModel, BatchedInferencePipeline
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
from tqdm import tqdm
import gc
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_processing.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize the model with GPU acceleration
try:
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")
    batched_model = BatchedInferencePipeline(model=model)
    logger.info("Successfully initialized Whisper model")
except Exception as e:
    logger.error(f"Failed to initialize Whisper model: {e}")
    sys.exit(1)

def process_audio_file(audio_path, language_code):
    """Process a single audio file and return transcription data."""
    try:
        # Validate input
        if not os.path.exists(audio_path):
            logger.warning(f"Audio file does not exist: {audio_path}")
            return False

        # Get the base filename without extension
        base_name = Path(audio_path).stem

        # Set the output directory for transcripts
        output_dir = Path("/mnt/f/Vojta/School/MSC_Thesis/WhisperVault/transcriptions")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{base_name}.json"

        # Check if transcription already exists in the output directory
        if output_file.exists():
            logger.info(f"Skipping {audio_path} - transcription already exists at {output_file}")
            return True

        # Check file size to avoid processing corrupted files
        file_size = os.path.getsize(audio_path)
        if file_size < 1024:  # Less than 1KB
            logger.warning(f"Skipping {audio_path} - file too small ({file_size} bytes)")
            return False

        # Transcribe with specified language
        try:
            segments, info = batched_model.transcribe(
                audio_path,
                batch_size=16,
                word_timestamps=True,
                language=language_code
            )
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            return False

        # Create a list to store all words with their timestamps
        words_data = []
        try:
            for segment in segments:
                if segment.words:
                    for word in segment.words:
                        words_data.append({
                            "word": word.word,
                            "start": round(word.start, 3),
                            "end": round(word.end, 3),
                            "confidence": round(word.probability, 3) if hasattr(word, 'probability') else None
                        })
        except Exception as e:
            logger.error(f"Error processing segments for {audio_path}: {e}")
            return False

        # Validate we have some data
        if not words_data:
            logger.warning(f"No words extracted from {audio_path}")
            return False

        # Write to JSON file in the specified output directory
        try:
            logger.info(f"Writing transcript to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                        "source_file": str(audio_path)
                    },
                    "words": words_data
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write transcription file {output_file}: {e}")
            return False

        logger.info(f"Successfully processed {audio_path} -> {output_file}")
        return True

    except Exception as e:
        logger.error(f"Unexpected error processing {audio_path}: {str(e)}")
        return False
    finally:
        # Force garbage collection to prevent memory issues
        gc.collect()

def process_language_folder(language_folder):
    """Process all audio files in a language folder."""
    try:
        if not os.path.exists(language_folder):
            logger.error(f"Language folder does not exist: {language_folder}")
            return
        
        language_code = os.path.basename(language_folder)
        audio_files = []
        
        # Find all mp3 and m4a files
        for ext in ['*.mp3', '*.m4a']:
            try:
                audio_files.extend(Path(language_folder).glob(ext))
            except Exception as e:
                logger.error(f"Error finding {ext} files in {language_folder}: {e}")
        
        if not audio_files:
            logger.info(f"No audio files found in {language_folder}")
            return
        
        logger.info(f"Processing {len(audio_files)} files in {language_folder}")
        
        # Process files with error handling
        successful = 0
        failed = 0
        
        # Process files in parallel using ThreadPoolExecutor with error handling
        with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers for safety
            futures = []
            for audio_file in audio_files:
                future = executor.submit(process_audio_file, str(audio_file), language_code)
                futures.append(future)
            
            # Process results with progress bar
            for future in tqdm(futures, desc=f"Processing {language_code}"):
                try:
                    result = future.result(timeout=300)  # 5 minute timeout per file
                    if result:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Future processing failed: {e}")
                    failed += 1
        
        logger.info(f"Completed {language_folder}: {successful} successful, {failed} failed")
        
    except Exception as e:
        logger.error(f"Error processing language folder {language_folder}: {e}")

def main():
    try:
        base_dir = Path("../PodcastRealDataset/rss_by_language")
        
        if not base_dir.exists():
            logger.error(f"Base directory does not exist: {base_dir}")
            sys.exit(1)
        
        # Get all language folders
        try:
            language_folders = [f for f in base_dir.iterdir() if f.is_dir()]
        except Exception as e:
            logger.error(f"Error reading language folders: {e}")
            sys.exit(1)
        
        if not language_folders:
            logger.warning("No language folders found")
            return
        
        logger.info(f"Found {len(language_folders)} language folders to process")
        
        # Process each language folder with error handling
        for folder in language_folders:
            try:
                logger.info(f"Starting processing of {folder}")
                process_language_folder(folder)
                logger.info(f"Completed processing of {folder}")
            except Exception as e:
                logger.error(f"Failed to process {folder}: {e}")
                continue  # Continue with next folder instead of crashing
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 