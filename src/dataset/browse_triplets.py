import os
import random
from src.dataset.make_dataset import AudioCognateDataset
from pydub import AudioSegment

DB_PATH = "/mnt/f/Vojta/School/MSC_Thesis/cognet_transfigure/cognates_2.db"
DATA_BASE_DIR = "data/processed"  # Where 'mel_spectrograms' directory will reside
N_MELS = 80
MAX_FRAMES = 500
MAX_LENGTH = None  # Or set to a value if you want to limit dataset size
AUDIO_CLIP_DIR = "triplet_audio_clips"
os.makedirs(AUDIO_CLIP_DIR, exist_ok=True)


def fetch_full_row(cursor, pron_id):
    cursor.execute("""
        SELECT p.*, w.text, w.language
        FROM pronunciations p
        JOIN words w ON w.word_id = CAST(substr(p.word_ids, 1, instr(p.word_ids || ',', ',') - 1) AS INTEGER)
        WHERE p.pron_id = ?
    """, (pron_id,))
    return cursor.fetchone()


def cut_and_save_audio(wave_path, time_start, time_end, out_path):
    # Remove the first '../' if present
    if wave_path.startswith("../"):
        wave_path = wave_path[3:]
    if not os.path.exists(wave_path):
        print(f"Audio file not found: {wave_path}")
        return
    try:
        audio = AudioSegment.from_file(wave_path)
        # pydub works in milliseconds
        start_ms = int(time_start * 1000)
        end_ms = int(time_end * 1000)
        segment = audio[start_ms:end_ms]
        segment.export(out_path, format="mp3")
    except Exception as e:
        print(f"Error processing {wave_path}: {e}")


def browse_random_triplets(dataset, n=50):
    dataset._connect_db()
    for i in range(n):
        anchor_comp_id = random.choice(dataset.eligible_component_ids)
        anchor_pron_ids_in_component = dataset.eligible_components[anchor_comp_id]
        anchor_pron_id, positive_pron_id = random.sample(anchor_pron_ids_in_component, 2)
        other_eligible_comp_ids = [cid for cid in dataset.eligible_component_ids if cid != anchor_comp_id]
        negative_comp_id = random.choice(other_eligible_comp_ids)
        negative_pron_ids_in_component = dataset.eligible_components[negative_comp_id]
        negative_pron_id = random.choice(negative_pron_ids_in_component)

        anchor_row = fetch_full_row(dataset.cursor, anchor_pron_id)
        positive_row = fetch_full_row(dataset.cursor, positive_pron_id)
        negative_row = fetch_full_row(dataset.cursor, negative_pron_id)

        print(f"\n=== Triplet {i+1} ===")
        print("Anchor:", anchor_row)
        print("Positive:", positive_row)
        print("Negative:", negative_row)

        # Unpack relevant fields: (pron_id, word_ids, transcription, lemma, wave_path, chopped_word_path, time_start, time_end, mel_path, text, language)
        for label, row in zip(["anchor", "positive", "negative"], [anchor_row, positive_row, negative_row]):
            if row is not None:
                wave_path = row[4]
                time_start = row[6]
                time_end = row[7]
                out_path = os.path.join(AUDIO_CLIP_DIR, f"triplet_{i+1}_{label}.mp3")
                cut_and_save_audio(wave_path, time_start, time_end, out_path)
    dataset._close_db()


if __name__ == "__main__":
    dataset = AudioCognateDataset(
        db_path=DB_PATH,
        data_base_dir=DATA_BASE_DIR,
        n_mels=N_MELS,
        max_frames=MAX_FRAMES,
        max_length=MAX_LENGTH
    )
    browse_random_triplets(dataset, n=50) 