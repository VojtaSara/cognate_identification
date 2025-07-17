import sqlite3
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import os
import time
import gc

# Constants
DB_PATH = "cognates_2.db"
OUTPUT_DIR = Path("mel_spectrograms")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
PADDING_MS = 83
MAX_DURATION_SEC = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GPU transforms once on startup
mel_transform = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=400, hop_length=160, n_mels=80).to(DEVICE)
db_transform = AmplitudeToDB().to(DEVICE)

def fetch_rows():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.wave_path, p.time_start, p.time_end, p.pron_id
        FROM pronunciations p
        JOIN words w ON w.word_id = CAST(substr(p.word_ids, 1, instr(p.word_ids || ',', ',') - 1) AS INTEGER)
        WHERE p.wave_path IS NOT NULL AND p.wave_path != '' AND (p.mel_path IS NULL OR p.mel_path = '')
        ORDER BY w.component_id, p.wave_path
    """)
    return cursor.fetchall()

def update_db(pron_id, mel_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE pronunciations SET mel_path = ? WHERE pron_id = ?", (str(mel_path), pron_id))
    conn.commit()
    conn.close()

def process(row):
    wave_path, start, end, pron_id = row
    try:
        if not os.path.exists(wave_path):
            print(f"❌ Not found: {wave_path}")
            return

        # Duration logic
        start_sec = max(0, start - PADDING_MS / 1000.0)
        duration = min(MAX_DURATION_SEC, (end - start) + 2 * PADDING_MS / 1000.0)
        frame_offset = int(start_sec * SAMPLE_RATE)
        num_frames = int(duration * SAMPLE_RATE)

        # Load
        waveform, sr = torchaudio.load(wave_path, frame_offset=frame_offset, num_frames=num_frames)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if waveform.shape[1] < 100:
            print(f"⚠️ Too short: {pron_id}")
            return

        waveform = waveform.to(DEVICE)
        mel = mel_transform(waveform)
        mel_db = db_transform(mel)

        out_path = OUTPUT_DIR / f"{pron_id}.pt"
        torch.save(mel_db.cpu(), out_path)
        update_db(pron_id, out_path)

        print(f"✅ {pron_id} saved.")
        del waveform, mel, mel_db
        gc.collect()

    except Exception as e:
        print(f"❌ Error on {pron_id}: {e}")

def main():
    rows = fetch_rows()
    print(f"Found {len(rows)} to process.")

    for i, row in enumerate(rows):
        print(f"[{i+1}/{len(rows)}]")
        process(row)

    print("✅ All done.")

if __name__ == "__main__":
    torch.set_num_threads(1)  # to avoid CPU overload in torchaudio
    main()
