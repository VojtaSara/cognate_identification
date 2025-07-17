import sqlite3
import torchaudio
import torch
import subprocess
import io
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import os
import gc
from collections import defaultdict

# Constants
DB_PATH = "cognates_3.db"
OUTPUT_DIR = Path("mel_spectrograms")
OUTPUT_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 16000
TARGET_SAMPLES = 8000  # 500 ms
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mel_transform = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=400,
    hop_length=160,
    n_mels=80
).to(DEVICE)

db_transform = AmplitudeToDB().to(DEVICE)

def load_audio_ffmpeg(path: str):
    cmd = [
        'ffmpeg', '-i', path,
        '-f', 'wav', '-ac', '1', '-ar', str(SAMPLE_RATE),
        '-loglevel', 'quiet', 'pipe:1'
    ]
    out = subprocess.check_output(cmd)
    waveform, _ = torchaudio.load(io.BytesIO(out))
    return waveform.squeeze(0)  # [T]

def fetch_all_unprocessed():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT p.wave_path, p.time_start, p.time_end, p.pron_id
        FROM pronunciations p
        JOIN words w ON w.word_id = CAST(substr(p.word_ids, 1, instr(p.word_ids || ',', ',') - 1) AS INTEGER)
        WHERE p.wave_path IS NOT NULL AND p.wave_path != '' AND (p.mel_path IS NULL OR p.mel_path = '')
        ORDER BY p.wave_path
    """)
    return cursor.fetchall()

def update_db(pron_id, mel_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE pronunciations SET mel_path = ? WHERE pron_id = ?", (str(mel_path), pron_id))
    conn.commit()
    conn.close()

def process_group(wave_path, rows, global_index_start):
    if not os.path.exists(wave_path):
        print(f"❌ File missing: {wave_path}")
        return 0

    try:
        waveform = load_audio_ffmpeg(wave_path).to(DEVICE)
        segment_data = []

        for (wave_path, t_start, t_end, pron_id) in rows:
            mid = (t_start + t_end) / 2
            start = max(0, mid - 0.25)
            i_start = int(start * SAMPLE_RATE)
            i_end = i_start + TARGET_SAMPLES

            if i_end > waveform.shape[0]:
                print(f"⚠️ Skipped (out of bounds): {pron_id}")
                continue

            segment = waveform[i_start:i_end]
            if segment.shape[0] < TARGET_SAMPLES:
                segment = torch.nn.functional.pad(segment, (0, TARGET_SAMPLES - segment.shape[0]))
            elif segment.shape[0] > TARGET_SAMPLES:
                segment = segment[:TARGET_SAMPLES]

            segment_data.append((pron_id, segment))

        if not segment_data:
            return 0

        segment_tensors = [seg.unsqueeze(0) for (_, seg) in segment_data]
        valid_rows = [(pron_id, 0, 0) for (pron_id, _) in segment_data]

        batch = torch.cat(segment_tensors, dim=0).to(DEVICE)  # [B, T]
        mel = mel_transform(batch)  # [B, 80, T]
        mel_db = db_transform(mel)

        for i, (pron_id, _, _) in enumerate(valid_rows):
            out_path = OUTPUT_DIR / f"{pron_id}.pt"
            torch.save(mel_db[i].cpu(), out_path)
            update_db(pron_id, out_path)
            print(f"✅ [{global_index_start+i}] {pron_id} saved.")

        del waveform, mel, mel_db
        gc.collect()
        return len(valid_rows)

    except Exception as e:
        print(f"❌ Error processing {wave_path}: {e}")
        return 0

def main():
    all_rows = fetch_all_unprocessed()
    groups = defaultdict(list)
    for row in all_rows:
        groups[row[0]].append(row)

    print(f"Found {len(all_rows)} entries across {len(groups)} audio files.")

    counter = 1
    for i, (wave_path, group_rows) in enumerate(groups.items()):
        print(f"\n▶ Processing [{i+1}/{len(groups)}]: {wave_path} ({len(group_rows)} segments)")
        processed = process_group(wave_path, group_rows, counter)
        counter += processed

    print("✅ All done.")

if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
