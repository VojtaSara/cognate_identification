import sqlite3
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback
import psutil
import os
import gc
import sys

# Configuration
DB_PATH = "cognates_2.db"
OUTPUT_DIR = Path("mel_spectrograms")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
PADDING_MS = 83

# WSL-safe settings
MAX_WORKERS = 2  # Reduced for WSL stability
BATCH_SIZE = 6   # Smaller batches for WSL
MEMORY_LIMIT_GB = 8  # Lower limit for WSL
PROCESS_TIMEOUT = 60  # Increased timeout

def check_wsl_environment():
    """Check if running in WSL and adjust settings accordingly"""
    is_wsl = "microsoft" in os.uname().release.lower() if hasattr(os, 'uname') else False
    if is_wsl:
        print("🐧 WSL environment detected - using conservative settings")
        return True
    return False

def check_system_resources():
    """Check if system has enough resources"""
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"System info:")
    print(f"  CPU cores: {cpu_count}")
    print(f"  Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
    print(f"  Memory usage: {memory.percent}%")
    
    if memory.percent > 90:  # More conservative for WSL
        print("⚠️  WARNING: High memory usage detected!")
        return False
    
    return True

def monitor_all_processes():
    """Monitor memory usage of all related processes"""
    current_process = psutil.Process()
    total_memory = current_process.memory_info().rss
    
    # Include child processes
    try:
        for child in current_process.children(recursive=True):
            try:
                total_memory += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    return total_memory / (1024 * 1024 * 1024)  # Convert to GB

def process_single_audio_safe(args):
    """Process a single audio file with WSL-safe settings"""
    wave_path, start, end, pron_id = args
    
    try:
        # Force CPU-only processing for stability
        torch.set_num_threads(1)  # Limit PyTorch threading
        
        # Check if file exists
        if not os.path.exists(wave_path):
            return {"success": False, "error": f"File not found: {wave_path}", "pron_id": pron_id}
        
        # Load audio with more conservative settings
        try:
            start_sec = max(0, start - PADDING_MS/1000.0)
            duration = (end - start) + 2 * PADDING_MS/1000.0
            
            frame_offset = int(start_sec * SAMPLE_RATE)
            num_frames = int(duration * SAMPLE_RATE)
            
            # Limit the number of frames to prevent memory issues
            max_frames = SAMPLE_RATE * 10  # Max 10 seconds
            if num_frames > max_frames:
                num_frames = max_frames
            
            waveform, sr = torchaudio.load(
                wave_path,
                frame_offset=frame_offset,
                num_frames=num_frames
            )
            
        except Exception as load_error:
            return {"success": False, "error": f"Audio loading failed: {str(load_error)}", "pron_id": pron_id}
        
        # Handle sample rate conversion
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
            del resampler  # Explicit cleanup
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        waveform = waveform.squeeze(0)
        
        if len(waveform) < 100:
            return {"success": False, "error": "Audio segment too short", "pron_id": pron_id}
        
        # Create transforms with explicit cleanup
        with torch.no_grad():
            mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)
            mel = mel_transform(waveform)
            del mel_transform  # Cleanup transform
            
            amp_to_db = AmplitudeToDB()
            mel_db = amp_to_db(mel)
            del amp_to_db  # Cleanup transform
            del mel  # Cleanup intermediate tensor
        
        # Save result
        out_path = OUTPUT_DIR / f"{pron_id}.pt"
        torch.save(mel_db, out_path)
        
        result = {
            "success": True, 
            "output_path": str(out_path), 
            "pron_id": pron_id,
            "shape": mel_db.shape
        }
        
        # Explicit cleanup
        del mel_db
        del waveform
        gc.collect()
        
        return result
        
    except Exception as e:
        # Ensure cleanup even on error
        gc.collect()
        return {
            "success": False, 
            "error": f"Unexpected error: {str(e)}", 
            "pron_id": pron_id,
            "traceback": traceback.format_exc()
        }

def batch_update_mel_paths(results):
    """Update database with results"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        updates = 0
        for result in results:
            if result.get("success") and result.get("output_path") and result.get("pron_id") is not None:
                cursor.execute(
                    "UPDATE pronunciations SET mel_path = ? WHERE pron_id = ? AND (mel_path IS NULL OR mel_path = '')",
                    (result["output_path"], result["pron_id"])
                )
                updates += cursor.rowcount
        
        conn.commit()
        conn.close()
        print(f"Database updated: {updates} records")
        
    except Exception as e:
        print(f"❌ Database update error: {str(e)}")

def process_batch_sequential_safe(rows):
    """Process batch sequentially with enhanced safety"""
    print("Processing sequentially (WSL-safe mode)...")
    results = []
    
    for i, row in enumerate(rows):
        print(f"Processing {i+1}/{len(rows)}: {row[3]}")
        
        # Process single item
        result = process_single_audio_safe(row)
        results.append(result)
        
        # Aggressive memory management
        gc.collect()
        
        # Check memory more frequently
        memory_gb = monitor_all_processes()
        if memory_gb > MEMORY_LIMIT_GB:
            print(f"⚠️  Memory limit reached ({memory_gb:.1f}GB), stopping batch")
            break
        
        # Brief pause to prevent overwhelming the system
        time.sleep(0.1)
    
    batch_update_mel_paths(results)
    return results

def process_batch_parallel_safe(rows, max_workers):
    """Process batch in parallel with WSL-safe settings"""
    print(f"Processing with {max_workers} workers (WSL-safe mode)...")
    results = []
    
    try:
        # Use spawn method for better WSL compatibility
        mp_context = mp.get_context('spawn')
        
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            # Submit jobs in smaller chunks
            chunk_size = min(max_workers * 2, len(rows))
            
            for chunk_start in range(0, len(rows), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(rows))
                chunk = rows[chunk_start:chunk_end]
                
                print(f"  Submitting chunk {chunk_start//chunk_size + 1} ({len(chunk)} files)")
                
                # Submit chunk
                future_to_row = {executor.submit(process_single_audio_safe, row): row for row in chunk}
                
                # Process completed jobs with timeout
                for i, future in enumerate(as_completed(future_to_row, timeout=PROCESS_TIMEOUT), 1):
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                        
                        if result["success"]:
                            print(f"✅ Chunk progress {i}/{len(chunk)}: Saved {result['pron_id']}")
                        else:
                            print(f"❌ Chunk progress {i}/{len(chunk)}: Failed {result['pron_id']} - {result['error']}")
                        
                    except Exception as e:
                        row = future_to_row[future]
                        print(f"❌ Worker error for {row[3]} - {str(e)}")
                        results.append({
                            "success": False, 
                            "error": f"Worker timeout/error: {str(e)}", 
                            "pron_id": row[3]
                        })
                
                # Memory check after each chunk
                memory_gb = monitor_all_processes()
                print(f"Memory after chunk: {memory_gb:.1f}GB")
                
                if memory_gb > MEMORY_LIMIT_GB:
                    print(f"⚠️  Memory limit reached, stopping")
                    break
                
                # Brief pause between chunks
                time.sleep(0.5)
                gc.collect()
                
    except Exception as e:
        print(f"❌ Parallel processing failed: {str(e)}")
        print("Falling back to sequential processing...")
        return process_batch_sequential_safe(rows)
    
    batch_update_mel_paths(results)
    return results

def process_entire_database_wsl_safe():
    """Main processing function optimized for WSL"""
    print("=== WSL-SAFE FULL DATABASE PROCESSING ===")
    
    # Check environment
    is_wsl = check_wsl_environment()
    
    # Adjust settings for WSL
    if is_wsl:
        global MAX_WORKERS, BATCH_SIZE, MEMORY_LIMIT_GB
        MAX_WORKERS = 1  # Single worker for maximum stability
        BATCH_SIZE = 3   # Very small batches
        MEMORY_LIMIT_GB = 2  # Conservative memory limit
        print(f"WSL adjustments: workers={MAX_WORKERS}, batch_size={BATCH_SIZE}, memory_limit={MEMORY_LIMIT_GB}GB")
    
    if not check_system_resources():
        print("❌ System resources insufficient, aborting.")
        return
    
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM pronunciations WHERE wave_path IS NOT NULL AND wave_path != '' AND (mel_path IS NULL OR mel_path = '')")
        total_count = cursor.fetchone()[0]
        print(f"Database: {total_count} pronunciation entries to process")
        
        if total_count == 0:
            print("❌ No unprocessed data found in database")
            return
        
        offset = 0
        processed = 0
        batch_num = 1
        start_time = time.time()
        
        while True:
            print(f"\n[DEBUG] Fetching batch {batch_num} at offset {offset}")
            
            cursor.execute("""
                SELECT wave_path, time_start, time_end, pron_id
                FROM pronunciations
                WHERE wave_path IS NOT NULL AND wave_path != '' AND (mel_path IS NULL OR mel_path = '')
                LIMIT ? OFFSET ?;
            """, (BATCH_SIZE, offset))
            
            rows = cursor.fetchall()
            if not rows:
                print(f"[DEBUG] No more rows to process")
                break
            
            print(f"--- Processing batch {batch_num} ({len(rows)} files, offset {offset}) ---")
            
            # Choose processing method based on environment
            if is_wsl or MAX_WORKERS == 1:
                batch_results = process_batch_sequential_safe(rows)
            else:
                workers = min(MAX_WORKERS, len(rows))
                batch_results = process_batch_parallel_safe(rows, workers)
            
            processed += len(batch_results)
            
            # Aggressive cleanup between batches
            gc.collect()
            
            memory_gb = monitor_all_processes()
            print(f"[DEBUG] Total memory usage after batch {batch_num}: {memory_gb:.1f}GB")
            
            if memory_gb > MEMORY_LIMIT_GB:
                print(f"⚠️  Memory limit reached, stopping processing")
                break
            
            offset += BATCH_SIZE
            batch_num += 1
            
            # Longer pause between batches for WSL
            if is_wsl:
                time.sleep(1.0)
        
        end_time = time.time()
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total processed: {processed}")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        print(f"Average time per file: {(end_time - start_time)/processed if processed else 0:.3f} seconds")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # Set multiprocessing start method for WSL compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Run WSL-safe processing
    process_entire_database_wsl_safe()