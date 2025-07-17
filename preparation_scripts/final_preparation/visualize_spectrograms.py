import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Directory containing .pt spectrograms
SPECTROGRAM_DIR = Path(__file__).parent / "mel_spectrograms"

# Only process specific files
target_files = ["16.pt", "20.pt"]

for fname in target_files:
    pt_file = SPECTROGRAM_DIR / fname
    if not pt_file.exists():
        print(f"Warning: {fname} does not exist.")
        continue
    try:
        if pt_file.stat().st_size == 0:
            print(f"Warning: {fname} is empty, skipping.")
            continue
        tensor = torch.load(pt_file)
        tensor = tensor.squeeze().cpu().numpy()
        plt.figure(figsize=(8, 5))
        plt.imshow(tensor, aspect='auto', origin='lower', cmap='magma')
        plt.title(fname)
        plt.axis('off')
        plt.tight_layout()
        out_path = SPECTROGRAM_DIR / f"spectrogram_{fname.replace('.pt', '.png')}"
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f"Saved: {out_path}")
    except Exception as e:
        print(f"Warning: Could not load {fname}: {e}") 