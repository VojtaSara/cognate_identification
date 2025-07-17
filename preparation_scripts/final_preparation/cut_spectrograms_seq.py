import sqlite3
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import os
import gc
import time

# Settings
DB_PATH = "cognates_3.db"
OUTPUT_DIR = Path("mel_spectrograms")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
PADDING_MS = 83

# Use more stable backend
torchaudio.set_audio_backend("sox_io")

def fetch_unprocessed_sorted():
    """Fetch unprocessed pronunciations sorted by cognate group (component_id), then audio file path."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Extract the first word_id from word_ids (comma-separated), join to words, and sort by component_id
    cursor.execute("""
        SELECT p.wave_path, p.time_start, p.time_end, p.pron_id
        FROM pronunciations p
        JOIN words w ON w.word_id = CAST(substr(p.word_ids, 1, instr(p.word_ids || ',', ',') - 1) AS INTEGER)
        WHERE p.wave_path IS NOT NULL AND p.wave_path != '' AND (p.mel_path IS NULL OR p.mel_path = '')
        ORDER BY w.component_id, p.wave_path
    """)
    
    rows = cursor.fetchall()
    conn.close()
    return rows

def update_db(pron_id, out_path):
    """Update database with mel_path for a single entry"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE pronunciations
        SET mel_path = ?
        WHERE pron_id = ?
    """, (str(out_path), pron_id))
    conn.commit()
    conn.close()

def process_audio(row):
    """Process a single row: load, extract mel, save."""
    wave_path, start, end, pron_id = row

    try:
        if not os.path.exists(wave_path):
            print(f"❌ File not found: {wave_path}")
            return False
        
        start_sec = max(0, start - PADDING_MS / 1000.0)
        duration = (end - start) + 2 * PADDING_MS / 1000.0

        frame_offset = int(start_sec * SAMPLE_RATE)
        num_frames = int(duration * SAMPLE_RATE)

        # Max 10s
        
        if num_frames > SAMPLE_RATE * 10:
            num_frames = SAMPLE_RATE * 10

        waveform, sr = torchaudio.load(wave_path, frame_offset=frame_offset, num_frames=num_frames)

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform.squeeze(0)

        if len(waveform) < 100:
            print(f"⚠️ Too short: {pron_id}")
            return False

        mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)(waveform)
        mel_db = AmplitudeToDB()(mel)

        out_path = OUTPUT_DIR / f"{pron_id}.pt"
        torch.save(mel_db.clone(), out_path)

        update_db(pron_id, out_path)

        print(f"✅ {pron_id} -> {out_path.name} [{mel_db.shape}]")
        del waveform, mel, mel_db
        gc.collect()
        return True

    except Exception as e:
        print(f"❌ Error on {pron_id}: {e}")
        return False

def main():
    print("=== SIMPLE MEL SPECTROGRAM CUT ===")
    start_time = time.time()
    rows = fetch_unprocessed_sorted()
    print(f"🔎 Found {len(rows)} items to process")

    processed = 0
    for i, row in enumerate(rows):
        print(f"\n[{i+1}/{len(rows)}] Processing...")
        if process_audio(row):
            processed += 1

    duration = time.time() - start_time
    print(f"\n=== DONE ===")
    print(f"Processed: {processed}/{len(rows)}")
    print(f"Time: {duration:.1f} sec | Avg/file: {duration / processed if processed else 0:.2f} sec")

if __name__ == "__main__":
    main()
