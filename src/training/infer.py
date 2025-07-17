import torch
import yaml
import numpy as np
from src.models.network import TransformerSiameseNet
from src.training.utils import set_device
from src.dataset.make_dataset import AudioCognateDataset
import torch.nn.functional as F

def load_model(model_path, config_path, device=None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if device is None:
        device = set_device()
    # Instantiate model
    model = TransformerSiameseNet(
        n_mels=config['model']['n_mels'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        embedding_dim=config['model']['embedding_dim']
    ).to(device)
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, device

def compute_similarity(spectrogram1, spectrogram2, model, device):
    """
    Compute similarity score between two spectrograms using a loaded model.
    Args:
        spectrogram1: np.ndarray or torch.Tensor, shape (n_mels, n_frames)
        spectrogram2: np.ndarray or torch.Tensor, shape (n_mels, n_frames)
        model: loaded model
        device: torch device
    Returns:
        similarity: float (cosine similarity between embeddings)
    """
    # Convert to torch tensors if needed
    if isinstance(spectrogram1, np.ndarray):
        spectrogram1 = torch.from_numpy(spectrogram1).float()
    if isinstance(spectrogram2, np.ndarray):
        spectrogram2 = torch.from_numpy(spectrogram2).float()
    # Add batch and channel dimensions: (1, 1, n_mels, n_frames)
    spectrogram1 = spectrogram1.unsqueeze(0).unsqueeze(0)
    spectrogram2 = spectrogram2.unsqueeze(0).unsqueeze(0)
    spectrogram1 = spectrogram1.to(device)
    spectrogram2 = spectrogram2.to(device)
    with torch.no_grad():
        emb1 = model._forward_one(spectrogram1)
        emb2 = model._forward_one(spectrogram2)
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(emb1, emb2).item()
        dist_pos = F.pairwise_distance(emb1, emb2, p=2).item()
    return similarity, dist_pos

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a random triplet from the dataset using a trained model.")
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create dataset
    dataset = AudioCognateDataset(
        db_path=config['data']['db_path'],
        data_base_dir=config['data']['data_base_dir'],
        n_mels=config['model']['n_mels'],
        split='test'
    )
    for i in range(10):
        # Get a random triplet (spectrogram, text)
        (anchor_spec, anchor_text, anchor_wav_path), (positive_spec, positive_text, positive_wav_path), (negative_spec, negative_text, negative_wav_path) = dataset.item_for_inference(0)

        model_path = "/mnt/f/Vojta/School/MSC_Thesis/experiments/exp_01_baseline/checkpoints/model_best_epoch_18_20250703_150520.pt"

        # Load model
        model, device = load_model(model_path, args.config)

        # Compute similarities
        sim_ap, dist_pos_ap = compute_similarity(anchor_spec, positive_spec, model, device)
        sim_an, dist_pos_an = compute_similarity(anchor_spec, negative_spec, model, device)

        print("Anchor text:", anchor_text)
        print("Positive text:", positive_text)
        print("Negative text:", negative_text)
        print("Anchor wav path:", anchor_wav_path)
        print("Positive wav path:", positive_wav_path)
        print("Negative wav path:", negative_wav_path)
        # print sum of each spectrogram
        print("Sum of anchor spectrogram:", anchor_spec.sum())
        print("Sum of positive spectrogram:", positive_spec.sum())
        print("Sum of negative spectrogram:", negative_spec.sum())
        print(f"Distance (anchor, positive): {dist_pos_ap:.4f}")
        print(f"Distance (anchor, negative): {dist_pos_an:.4f}")