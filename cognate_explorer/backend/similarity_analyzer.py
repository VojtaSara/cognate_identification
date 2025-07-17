import torch
import torchaudio
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from pathlib import Path
import json
import os
import gc
import sys
import yaml
import numpy as np
from typing import List, Dict, Tuple

# Add the src directory to the path to import the model
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

try:
    from models.network import TransformerSiameseNet
    from training.utils import set_device
except ImportError as e:
    print(f"Error importing model modules: {e}")
    print(f"Added to sys.path: {src_path}")
    print(f"Available modules: {[p.name for p in src_path.iterdir() if p.is_dir()]}")
    raise

SAMPLE_RATE = 16000
PADDING_MS = 83
MEL_DIR = Path("mel_spectrograms")
MEL_DIR.mkdir(exist_ok=True)

class SimilarityAnalyzer:
    def __init__(self, model_path: str, config_path: str):
        """Initialize the similarity analyzer with a trained model."""
        # Force CPU for now due to RTX 5060 Ti compatibility issues
        self.device = torch.device("cpu")
        print(f"[INFO] Using device: {self.device}")
        self.model, self.device = self._load_model(model_path, config_path)
        
    def _load_model(self, model_path: str, config_path: str):
        """Load the trained model."""
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Instantiate model
        model = TransformerSiameseNet(
            n_mels=config['model']['n_mels'],
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            embedding_dim=config['model']['embedding_dim']
        ).to(self.device)
        
        # Load weights - force CPU loading
        print(f"[INFO] Loading model from {model_path} to {self.device}")
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"[INFO] Model loaded successfully on {self.device}")
        return model, self.device
    
    def cut_mel_segments(self, audio_path: str, words: List[Dict]) -> List[str]:
        """Cut mel spectrograms for each word in the transcript."""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            waveform = waveform.squeeze(0)
        except Exception as e:
            print(f"❌ Error loading audio: {e}")
            return []

        mel_paths = []
        for idx, word in enumerate(words):
            start = max(0, word["start"] - PADDING_MS / 1000.0)
            duration = (word["end"] - word["start"]) + 2 * PADDING_MS / 1000.0
            frame_offset = int(start * SAMPLE_RATE)
            num_frames = int(duration * SAMPLE_RATE)
            
            # Max 10s
            if num_frames > SAMPLE_RATE * 10:
                num_frames = SAMPLE_RATE * 10
                
            segment = waveform[frame_offset:frame_offset+num_frames]
            if len(segment) < 100:
                mel_paths.append(None)
                continue
                
            mel = MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=80)(segment)
            mel_db = AmplitudeToDB()(mel)
            mel_path = MEL_DIR / f"word_{idx}.pt"
            torch.save(mel_db.clone(), mel_path)
            mel_paths.append(str(mel_path))
            del mel, mel_db, segment
            gc.collect()
            
        return mel_paths
    
    def compute_similarity(self, spectrogram1: torch.Tensor, spectrogram2: torch.Tensor) -> Tuple[float, float]:
        """
        Compute similarity score between two spectrograms using the loaded model.
        Returns: (cosine_similarity, euclidean_distance)
        """
        # Convert to torch tensors if needed
        if isinstance(spectrogram1, np.ndarray):
            spectrogram1 = torch.from_numpy(spectrogram1).float()
        if isinstance(spectrogram2, np.ndarray):
            spectrogram2 = torch.from_numpy(spectrogram2).float()
            
        # Add batch and channel dimensions: (1, 1, n_mels, n_frames)
        spectrogram1 = spectrogram1.unsqueeze(0).unsqueeze(0)
        spectrogram2 = spectrogram2.unsqueeze(0).unsqueeze(0)
        spectrogram1 = spectrogram1.to(self.device)
        spectrogram2 = spectrogram2.to(self.device)
        
        with torch.no_grad():
            emb1 = self.model._forward_one(spectrogram1)
            emb2 = self.model._forward_one(spectrogram2)
            # Compute cosine similarity
            similarity = F.cosine_similarity(emb1, emb2).item()
            dist_pos = F.pairwise_distance(emb1, emb2, p=2).item()
            
        return similarity, dist_pos
    
    def analyze_transcript_similarity(self, transcript_json_path: str, audio_path: str) -> Dict:
        """
        Analyze similarity between all words in a transcript.
        Uses the first word as the reference and compares it with all others.
        """
        # Load transcript
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        words = data["words"] if "words" in data else data.get("result", {}).get("words", [])
        
        if len(words) < 2:
            return {"error": "Transcript must have at least 2 words for similarity analysis"}
        
        # Cut mel spectrograms for all words
        print(f"Cutting mel spectrograms for {len(words)} words...")
        mel_paths = self.cut_mel_segments(audio_path, words)
        
        # Update transcript with mel paths
        for word, mel_path in zip(words, mel_paths):
            word["mel_path"] = mel_path
        
        # Load the first spectrogram as reference
        if mel_paths[0] is None:
            return {"error": "First word spectrogram could not be created"}
            
        reference_spec = torch.load(mel_paths[0])
        
        # Compare reference with all other spectrograms
        similarities = []
        for idx, (word, mel_path) in enumerate(zip(words, mel_paths)):
            if idx == 0 or mel_path is None:
                continue
                
            try:
                current_spec = torch.load(mel_path)
                similarity, distance = self.compute_similarity(reference_spec, current_spec)
                
                similarities.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "similarity_score": similarity,
                    "distance": distance,
                    "mel_path": mel_path
                })
            except Exception as e:
                print(f"Error processing word {idx}: {e}")
                similarities.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "similarity_score": None,
                    "distance": None,
                    "mel_path": mel_path,
                    "error": str(e)
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity_score"] if x["similarity_score"] is not None else -1, reverse=True)
        
        # Save updated transcript with mel paths
        updated_json_path = Path(transcript_json_path).with_name(Path(transcript_json_path).stem + "_with_similarity.json")
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "reference_word": words[0]["word"],
            "reference_mel_path": mel_paths[0],
            "similarities": similarities,
            "updated_transcript_path": str(updated_json_path)
        }
    
    def analyze_transcript_with_reference(self, transcript_json_path: str, audio_path: str, reference_spec_path: str) -> Dict:
        """
        Analyze similarity between a reference spectrogram and all words in a transcript.
        """
        # Load transcript
        with open(transcript_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        words = data["words"] if "words" in data else data.get("result", {}).get("words", [])
        
        if len(words) < 1:
            return {"error": "Transcript must have at least 1 word for similarity analysis"}
        
        # Cut mel spectrograms for all words
        print(f"Cutting mel spectrograms for {len(words)} words...")
        mel_paths = self.cut_mel_segments(audio_path, words)
        
        # Update transcript with mel paths
        for word, mel_path in zip(words, mel_paths):
            word["mel_path"] = mel_path
        
        # Load the reference spectrogram
        if not Path(reference_spec_path).exists():
            return {"error": f"Reference spectrogram not found: {reference_spec_path}"}
            
        reference_spec = torch.load(reference_spec_path)
        
        # Compare reference with all word spectrograms
        similarities = []
        for idx, (word, mel_path) in enumerate(zip(words, mel_paths)):
            if mel_path is None:
                continue
                
            try:
                current_spec = torch.load(mel_path)
                similarity, distance = self.compute_similarity(reference_spec, current_spec)
                
                similarities.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "similarity_score": similarity,
                    "distance": distance,
                    "mel_path": mel_path
                })
            except Exception as e:
                print(f"Error processing word {idx}: {e}")
                similarities.append({
                    "word": word["word"],
                    "start": word["start"],
                    "end": word["end"],
                    "similarity_score": None,
                    "distance": None,
                    "mel_path": mel_path,
                    "error": str(e)
                })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity_score"] if x["similarity_score"] is not None else -1, reverse=True)
        
        # Save updated transcript with mel paths
        updated_json_path = Path(transcript_json_path).with_name(Path(transcript_json_path).stem + "_with_reference_similarity.json")
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return {
            "reference_spec_path": reference_spec_path,
            "similarities": similarities,
            "updated_transcript_path": str(updated_json_path)
        }

def add_similarity_analysis_to_transcript(json_path: str, audio_path: str, model_path: str, config_path: str) -> Dict:
    """
    Main function to add similarity analysis to a transcript.
    Returns the analysis results.
    """
    analyzer = SimilarityAnalyzer(model_path, config_path)
    return analyzer.analyze_transcript_similarity(json_path, audio_path)

def add_similarity_analysis_with_reference(json_path: str, audio_path: str, reference_spec_path: str, model_path: str, config_path: str) -> Dict:
    """
    Main function to add similarity analysis using a reference spectrogram.
    Returns the analysis results.
    """
    analyzer = SimilarityAnalyzer(model_path, config_path)
    return analyzer.analyze_transcript_with_reference(json_path, audio_path, reference_spec_path)

# CLI usage
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python similarity_analyzer.py <transcript.json> <audiofile> <model_path> <config_path>")
        sys.exit(1)
    
    result = add_similarity_analysis_to_transcript(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print(json.dumps(result, indent=2)) 