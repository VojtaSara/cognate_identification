import torch
import matplotlib.pyplot as plt
from pathlib import Path

MEL_DIR = Path("mel_spectrograms")

def show_mel_spectrograms(pron_ids):
    mels = []
    valid_ids = []
    for pron_id in pron_ids:
        mel_path = MEL_DIR / f"{pron_id}.pt"
        if not mel_path.exists():
            print(f"File not found: {mel_path}")
            continue
        mel = torch.load(mel_path)
        mels.append(mel)
        valid_ids.append(pron_id)
    if not mels:
        print("No valid mel spectrograms to display.")
        return
    n = len(mels)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 4))
    if n == 1:
        axes = [axes]
    for ax, mel, pron_id in zip(axes, mels, valid_ids):
        im = ax.imshow(mel.numpy(), aspect='auto', origin='lower')
        ax.set_title(f"Pron ID: {pron_id}")
        ax.set_xlabel('Frame')
        ax.set_ylabel('Mel Bin')
        fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example usage: replace with your own list of pron_ids
    pron_ids = [57795, 118278, 6986, 56565]
    show_mel_spectrograms(pron_ids)