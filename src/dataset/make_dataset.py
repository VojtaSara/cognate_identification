# src/dataset/make_dataset.py
import torch
import torch.nn.functional as F
import torchaudio
import os
import sqlite3
import random
from typing import Tuple, Dict, List
from torch.utils.data import DataLoader


class AudioCognateDataset:
    """
    Dataset class for loading audio spectrograms from a SQLite database
    and generating triplets (anchor, positive, negative) for triplet loss.

    It connects to a SQLite database with the following schema:
    - cognate_groups (component_id, concept_id, description)
    - words (word_id, component_id, text, language)
    - pronunciations (pron_id, word_ids, transcription, lemma, wave_path, chopped_word_path, time_start, time_end, mel_path)

    A triplet consists of:
    - Anchor: A pronunciation (pron_id).
    - Positive: Another pronunciation from the SAME cognate component as the anchor.
    - Negative: A pronunciation from a DIFFERENT cognate component than the anchor.

    Only pronunciations with a valid 'mel_path' are considered for training.
    """
    def __init__(self, db_path: str, data_base_dir: str, n_mels: int = 80, max_frames: int = 500, split: str = "train", dataset_size_factor: float = 1, use_specaugment: bool = False):
        """
        Args:
            db_path (str): Path to the SQLite database file (e.g., 'data/metadata/cognates_2.db').
            data_base_dir (str): Base directory where mel spectrogram files are stored.
                                  The 'mel_path' in the DB is expected to be relative to this.
                                  (e.g., if mel_path is 'mel_spectrograms/3.pt', then the full path
                                  will be os.path.join(data_base_dir, 'mel_spectrograms/3.pt')).
            n_mels (int): Expected number of mel bins in the spectrograms. Used for validation.
            max_frames (int): Maximum number of frames for padding/truncation of spectrograms.
            split (str): Dataset split ('train', 'test', 'all').
            dataset_size_factor (float): Factor to reduce dataset size (1.0 = full dataset).
            use_specaugment (bool): Whether to apply SpecAugment during training.
        """
        self.db_path = db_path
        self.data_base_dir = data_base_dir
        self.n_mels = n_mels
        self.max_frames = max_frames
        self.dataset_size_factor = dataset_size_factor
        self.split = split
        self.use_specaugment = use_specaugment
        self.conn = None
        self.cursor = None

        # Data structures to store mappings after loading from DB
        self.component_to_prons: Dict[str, List[str]] = {} # Maps component_id to a list of pron_ids
        self.pron_to_mel_path: Dict[str, str] = {}         # Maps pron_id to its full mel_path
        self.pron_to_component: Dict[str, str] = {}        # Maps pron_id to its component_id
        self.pron_to_language: Dict[str, str] = {}        # Maps pron_id to its language
        self.pron_to_text: Dict[str, str] = {}            # Maps pron_id to its text
        self.pron_to_wav_path: Dict[str, str] = {}        # Maps pron_id to its wav_path

        # Initialize SpecAugment if conditions are met
        self.spec_augment = None
        if split == "train" and use_specaugment:
            self.spec_augment = torch.nn.Sequential(
                torchaudio.transforms.FrequencyMasking(freq_mask_param=15),
                torchaudio.transforms.TimeMasking(time_mask_param=35)
            )

        self._connect_db()
        self.component_to_prons = self._load_pronunciation_cognate_mappings(split)
        self._close_db() # Close after loading data into memory

        # Filter out components that don't have enough pronunciations to form a positive pair
        self.eligible_components = {
            comp_id: pron_ids for comp_id, pron_ids in self.component_to_prons.items() if len(pron_ids) >= 2
        }
        self.eligible_component_ids = list(self.eligible_components.keys())

        if not self.eligible_component_ids:
            raise ValueError(
                "No eligible cognate components found in the database. "
                "Ensure mel_path is present and components have at least 2 pronunciations each."
            )
        print(f"Dataset initialized with {len(self.eligible_component_ids)} eligible cognate components.")
        if self.spec_augment:
            print("SpecAugment enabled for training data augmentation.")


    def _connect_db(self):
        """Establishes a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            print(f"Database connection error: {e}")
            raise

    def _close_db(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
            print("Database connection closed.")

    def _load_pronunciation_cognate_mappings(self, split: str = "train"):
        """
        Loads mappings from pronunciations to cognate components, mel_paths, and languages.
        Filters out pronunciations without a valid mel_path.
        Supports split: 'train', 'test', or 'all'.
        """
        query = """
            SELECT
                p.pron_id,
                p.mel_path,
                cg.component_id,
                w.language,
                w.text,
                p.wave_path
            FROM
                pronunciations p
            JOIN
                words w ON p.word_ids = CAST(w.word_id AS TEXT)
            JOIN
                cognate_groups cg ON w.component_id = cg.component_id
            WHERE
                p.mel_path IS NOT NULL AND p.mel_path != ''
        """

        # Append filtering based on split
        if split == "train":
            query += " AND cg.TRAIN_test_split = 0"
        elif split == "test":
            query += " AND cg.TRAIN_test_split = 1"
        elif split == "all":
            pass  # No additional filtering
        else:
            raise ValueError(f"Unknown split type: {split!r}. Use 'train', 'test', or 'all'.")

        # Execute query (example, assuming self.conn is a valid SQLite connection)
        cursor = self.conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        component_to_prons = {}
        for pron_id, mel_path, component_id, language, text, wav_path in rows:
            if component_id not in component_to_prons:
                component_to_prons[component_id] = []
            component_to_prons[component_id].append(pron_id)
            # Fill self.pron_to_mel_path and self.pron_to_component here
            self.pron_to_mel_path[pron_id] = os.path.join(self.data_base_dir, mel_path)
            self.pron_to_component[pron_id] = component_id
            self.pron_to_language[pron_id] = language
            self.pron_to_text[pron_id] = text
            self.pron_to_wav_path[pron_id] = wav_path
        

        if self.dataset_size_factor < 1 and split == "train":
            # todo: reduce keys by dataset_size_factor
            keys = list(component_to_prons.keys())
            random.shuffle(keys)
            keys = keys[:int(len(keys) * self.dataset_size_factor)]
            component_to_prons = {k: component_to_prons[k] for k in keys}

        return component_to_prons


    def _load_spectrogram(self, full_mel_path: str) -> torch.Tensor:
        """Loads a .pt spectrogram file and pads/truncates it."""
        if not os.path.exists(full_mel_path):
            raise FileNotFoundError(f"Spectrogram file not found: {full_mel_path}")
        
        spectrogram = torch.load(full_mel_path)
        
        # Ensure spectrogram has 2 dimensions (n_mels, n_frames)
        # It might be loaded as [1, n_mels, n_frames] if saved with a channel dim
        if spectrogram.dim() == 3:
            spectrogram = spectrogram.squeeze(0)
        elif spectrogram.dim() != 2:
            raise ValueError(f"Unexpected spectrogram dimension: {spectrogram.dim()} at {full_mel_path}. Expected 2 or 3.")
        
        # Pad or truncate to max_frames
        n_frames = spectrogram.shape[1]
        if n_frames < self.max_frames:
            pad_width = self.max_frames - n_frames
            # Pad along the time dimension (dimension 1). F.pad expects (left, right, top, bottom) for 2D.
            spectrogram = F.pad(spectrogram, (0, pad_width))
        elif n_frames > self.max_frames:
            spectrogram = spectrogram[:, :self.max_frames]
        
        # Ensure correct mel dimension
        if spectrogram.shape[0] != self.n_mels:
            raise ValueError(
                f"Spectrogram n_mels mismatch for {full_mel_path}. "
                f"Expected {self.n_mels}, got {spectrogram.shape[0]}."
            )
        return spectrogram

    def __len__(self):
        """
        Returns the number of eligible anchors. An anchor is any pronunciation
        that belongs to a cognate component with at least two pronunciations.
        """
        # This count represents the total number of individual pronunciations
        # that can serve as an anchor in a triplet, ensuring a positive can always be found.
        count = 0
        for comp_id in self.eligible_component_ids:
            count += len(self.eligible_components[comp_id])


        #return count

        # This is the full dataset size, for constant epoch length
        return 125440

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a triplet (anchor_spec, positive_spec, negative_spec) for the given idx.
        The idx is mapped to a specific anchor pronunciation, and positive/negative are chosen.
        Implements harder triplet mining: positive is from a different language if possible.
        """
        # 1. Choose a random cognate component for the anchor (must have >= 2 prons)
        anchor_comp_id = random.choice(self.eligible_component_ids)
        anchor_pron_ids_in_component = self.eligible_components[anchor_comp_id]
        anchor_pron_id = random.choice(anchor_pron_ids_in_component)
        anchor_lang = self.pron_to_language[anchor_pron_id]
        # Try to pick a positive from a different language
        positive_candidates = [pid for pid in anchor_pron_ids_in_component if self.pron_to_language[pid] != anchor_lang]
        if not positive_candidates:
            # if no positive candidates in different language, fail the entire triplet (HARDER MINING)
            positive_candidates = [pid for pid in anchor_pron_ids_in_component if pid != anchor_pron_id]
        if not positive_candidates:
            # If still empty (shouldn't happen if eligible_components is correct), fail the entire triplet (HARDER MINING)
            positive_pron_id = anchor_pron_id
        else:
            positive_pron_id = random.choice(positive_candidates)

        # 2. Choose a negative pron_id from a different cognate component
        other_eligible_comp_ids = [cid for cid in self.eligible_component_ids if cid != anchor_comp_id]

        if not other_eligible_comp_ids:
            raise ValueError(
                "Not enough distinct cognate components with >=2 pronunciations to form a negative pair. "
                "The dataset requires at least two such components."
            )
        negative_comp_id = random.choice(other_eligible_comp_ids)
        negative_pron_ids_in_component = self.eligible_components[negative_comp_id] 
        negative_pron_id = random.choice(negative_pron_ids_in_component)

        # 3. Get mel_paths
        anchor_mel_path = self.pron_to_mel_path[anchor_pron_id]
        positive_mel_path = self.pron_to_mel_path[positive_pron_id]
        negative_mel_path = self.pron_to_mel_path[negative_pron_id]

        # 4. Load spectrograms
        anchor_spec = self._load_spectrogram(anchor_mel_path)
        positive_spec = self._load_spectrogram(positive_mel_path)
        negative_spec = self._load_spectrogram(negative_mel_path)
        
        # 5. Apply SpecAugment if enabled for training
        if self.spec_augment:
            # Add channel dimension for SpecAugment (expects [C, F, T])
            anchor_spec = anchor_spec.unsqueeze(0)  # [1, F, T]
            positive_spec = positive_spec.unsqueeze(0)  # [1, F, T]
            negative_spec = negative_spec.unsqueeze(0)  # [1, F, T]
            
            # Apply SpecAugment
            anchor_spec = self.spec_augment(anchor_spec)
            positive_spec = self.spec_augment(positive_spec)
            negative_spec = self.spec_augment(negative_spec)
            
            # Remove channel dimension to return [F, T]
            anchor_spec = anchor_spec.squeeze(0)
            positive_spec = positive_spec.squeeze(0)
            negative_spec = negative_spec.squeeze(0)
        
        return anchor_spec, positive_spec, negative_spec

    def item_for_inference(self, idx: int):
        """
        Generates a triplet for inference: each is a tuple (spectrogram, text, wav_path).
        Ensures that the positive sample is a different word and comes from a different language than the anchor.
        """
        # 1. Choose a random cognate component for the anchor (must have >= 2 prons)
        anchor_comp_id = random.choice(self.eligible_component_ids)
        anchor_pron_ids_in_component = self.eligible_components[anchor_comp_id]
        anchor_pron_id = random.choice(anchor_pron_ids_in_component)
        anchor_lang = self.pron_to_language[anchor_pron_id]
        # Try to pick a positive from a different language
        positive_candidates = [pid for pid in anchor_pron_ids_in_component if self.pron_to_language[pid] != anchor_lang]
        if not positive_candidates:
            # fallback: just pick any other (old behavior)
            #positive_candidates = [pid for pid in anchor_pron_ids_in_component if pid != anchor_pron_id]
            return None
        #if not positive_candidates:
            # If still empty (shouldn't happen if eligible_components is correct), fallback to anchor itself
        #    positive_pron_id = anchor_pron_id
        #else:
        positive_pron_id = random.choice(positive_candidates)

        # 2. Choose a negative pron_id from a different cognate component
        other_eligible_comp_ids = [cid for cid in self.eligible_component_ids if cid != anchor_comp_id]

        if not other_eligible_comp_ids:
            raise ValueError(
                "Not enough distinct cognate components with >=2 pronunciations to form a negative pair. "
                "The dataset requires at least two such components."
            )
        negative_comp_id = random.choice(other_eligible_comp_ids)
        negative_pron_ids_in_component = self.eligible_components[negative_comp_id] 
        negative_pron_id = random.choice(negative_pron_ids_in_component)

        # 3. Get mel_paths
        anchor_mel_path = self.pron_to_mel_path[anchor_pron_id]
        positive_mel_path = self.pron_to_mel_path[positive_pron_id]
        negative_mel_path = self.pron_to_mel_path[negative_pron_id]

        # 4. Load spectrograms
        anchor_spec = self._load_spectrogram(anchor_mel_path)
        positive_spec = self._load_spectrogram(positive_mel_path)
        negative_spec = self._load_spectrogram(negative_mel_path)

        # 5. Get text, wav_path, and language
        anchor_text, anchor_wav_path, _ = self.pron_to_text[anchor_pron_id], self.pron_to_wav_path[anchor_pron_id], self.pron_to_language[anchor_pron_id]
        positive_text, positive_wav_path, _ = self.pron_to_text[positive_pron_id], self.pron_to_wav_path[positive_pron_id], self.pron_to_language[positive_pron_id]
        negative_text, negative_wav_path, _ = self.pron_to_text[negative_pron_id], self.pron_to_wav_path[negative_pron_id], self.pron_to_language[negative_pron_id]
        return (anchor_spec, anchor_text, anchor_wav_path), (positive_spec, positive_text, positive_wav_path), (negative_spec, negative_text, negative_wav_path)



if __name__ == '__main__':
    # --- Example Usage ---
    # Define paths based on the project structure
    DB_PATH = "data/metadata/cognates_2.db"
    DATA_BASE_DIR = "data/processed" # Where 'mel_spectrograms' directory will reside

    # This script now expects the database and spectrogram files to already exist.
    # If they don't, you will need to create them manually or via another script.
    print("--- Testing AudioCognateDataset ---")

    # 2. Initialize the dataset
    try:
        dataset = AudioCognateDataset(db_path=DB_PATH, data_base_dir=DATA_BASE_DIR, n_mels=80, max_frames=500)
        
        print(f"Total eligible pronunciations (potential anchors): {len(dataset)}")

        # 3. Test fetching a single triplet
        if len(dataset) > 0:
            print("\nFetching a sample triplet:")
            anchor_spec, positive_spec, negative_spec = dataset[0] # Use dataset[0] to fetch one triplet
            print(f"Anchor spec shape: {anchor_spec.shape}")
            print(f"Positive spec shape: {positive_spec.shape}")
            print(f"Negative spec shape: {negative_spec.shape}")
        else:
            print("Dataset is empty after filtering. Cannot fetch triplets.")

        # 4. Test with DataLoader
        BATCH_SIZE = 2
        # num_workers=0 is used here for simpler debugging. For performance, use higher values.
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 

        print(f"\nIterating through {min(2, len(dataloader))} batches (Batch Size: {BATCH_SIZE}):")
        for i, (anchor_batch, positive_batch, negative_batch) in enumerate(dataloader):
            print(f"\nBatch {i+1}:")
            print(f"Anchor batch shape: {anchor_batch.shape}")
            print(f"Positive batch shape: {positive_batch.shape}")
            print(f"Negative batch shape: {negative_batch.shape}")
            if i >= 1: # Just show a couple of batches
                break
        
    except Exception as e:
        print(f"An error occurred during dataset initialization or testing: {e}")

