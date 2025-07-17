# src/training/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import yaml
from datetime import datetime
import argparse
import wandb
import numpy as np
# Import modules from your project
from src.models.network import TransformerSiameseNet
from src.training.utils import TripletLoss, save_checkpoint, load_checkpoint, set_device
from src.dataset.make_dataset import AudioCognateDataset # No longer importing create_dummy_data
import tqdm

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm.tqdm(dataloader, total=len(dataloader), desc="Evaluating"):
            anchor = anchor.to(device).unsqueeze(1)
            positive = positive.to(device).unsqueeze(1)
            negative = negative.to(device).unsqueeze(1)
            anchor_emb = model._forward_one(anchor)
            positive_emb = model._forward_one(positive)
            negative_emb = model._forward_one(negative)
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss

# ROC AUC computation removed to lighten training load

def train(config_path: str):
    """
    Main training function for the Transformer Siamese Network.

    This function performs the following steps:
    1. Loads training configuration from a YAML file.
    2. Sets up the computation device (GPU if available, else CPU).
    3. Initializes the dataset and DataLoader for generating triplets.
    4. Initializes the Transformer Siamese model, Triplet Loss criterion, and optimizer.
    5. Optionally loads a checkpoint to resume training.
    6. Executes the training loop for a specified number of epochs.
    7. Saves the best model checkpoint based on training loss.

    Args:
        config_path (str): Path to the YAML configuration file for training parameters.
    """
    # 1. Load configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}. Please ensure it exists.")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing config file {config_path}: {e}")
        return

    # 2. Setup wandb
    run = wandb.init(project="audio-cognate-detection", config=config)

    # 2. Setup device
    device = set_device()

    # --- 3. Dataset and DataLoader ---
    db_path = config['data']['db_path']
    data_base_dir = config['data']['data_base_dir'] 
    
    # Check if database and spectrogram directory exist.
    # The script now *expects* these to be present.
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at {db_path}. Please ensure your database is ready.")
        return
    mel_spectrograms_dir_full_path = os.path.join(data_base_dir, 'mel_spectrograms')
    if not os.path.exists(mel_spectrograms_dir_full_path):
        print(f"Error: Mel spectrograms directory not found at {mel_spectrograms_dir_full_path}. "
              "Please ensure your spectrograms are processed and stored correctly.")
        return


    # Load dataset
    dataset_train = AudioCognateDataset(
        db_path=db_path,
        data_base_dir=data_base_dir,
        n_mels=config['model']['n_mels'],
        max_frames=config['data']['max_frames'],
        split = "train",
        dataset_size_factor=config['data'].get('dataset_size_factor', 1.0),
        use_specaugment=config['data'].get('use_specaugment', False)
    )
    dataset_val = AudioCognateDataset(
        db_path=db_path,
        data_base_dir=data_base_dir,
        n_mels=config['model']['n_mels'],
        max_frames=config['data']['max_frames'],
        split = "test",
        use_specaugment=False  # No augmentation for validation
    )
    
    print(f"Dataset size: {len(dataset_train)}")
    if len(dataset_train) == 0:
        print("Error: Dataset is empty after filtering. No eligible triplets can be formed for training.")
        print("Please ensure your database contains cognate components with at least two pronunciations each, and valid mel_paths.")
        return


    # Create DataLoaders for each dataset
    train_loader = DataLoader(
        dataset_train,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    val_loader = DataLoader(
        dataset_val,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )

    print(f"Dataset loaded with {len(dataset_train)} eligible triplets (potential anchors).")
    print(f"DataLoader prepared with batch size {config['training']['batch_size']}.")

    print(f"Validation dataset loaded with {len(dataset_val)} eligible triplets (potential anchors).")
    print(f"Validation DataLoader prepared with batch size {config['training']['batch_size']}.")

    # --- 4. Model, Loss, Optimizer ---
    model = TransformerSiameseNet(
        n_mels=config['model']['n_mels'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        embedding_dim=config['model']['embedding_dim']
    ).to(device)

    # print the model architecture
    print(model)

    criterion = TripletLoss(margin=config['loss']['margin'])
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # --- 5. Checkpoint Loading (Optional) ---
    start_epoch = 1
    best_loss = float('inf')
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists
    
    if config['training'].get('resume_checkpoint', False):
        resume_checkpoint_name = config['training'].get('resume_checkpoint_name')
        if resume_checkpoint_name:
            checkpoint_path = os.path.join(checkpoint_dir, resume_checkpoint_name)
            if os.path.exists(checkpoint_path):
                start_epoch, best_loss = load_checkpoint(checkpoint_path, model, optimizer)
            else:
                print(f"Specified resume checkpoint '{checkpoint_path}' not found. Starting training from scratch.")
        else:
            print("resume_checkpoint is True, but resume_checkpoint_name is not specified. Starting from scratch.")

    
    val_loss_initial = evaluate(model, val_loader, criterion, device)
    train_loss_initial = evaluate(model, train_loader, criterion, device)
    
    run.log({
        "train_loss": train_loss_initial,
        "val_loss": val_loss_initial,
        "epoch": 0,
    })

    print(f"Training Loss: {train_loss_initial:.6f}")
    print(f"Validation Loss: {val_loss_initial:.6f}")
    

    # --- 6. Training Loop ---
    print("\n--- Starting Training ---")
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Iterate over batches from the DataLoader
        for batch_idx, (anchor, positive, negative) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader), desc="Training"):
            # Move tensors to the specified device (GPU/CPU)
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # (B,M,N) -> (B,1,M,N)
            anchor = anchor.unsqueeze(1)
            positive = positive.unsqueeze(1)
            negative = negative.unsqueeze(1)

            optimizer.zero_grad() # Clear gradients from previous step
            
            # Forward pass: get embeddings for anchor, positive, and negative samples
            # _forward_one is called for each input separately since it's a Siamese network
            anchor_emb = model._forward_one(anchor)
            positive_emb = model._forward_one(positive)
            negative_emb = model._forward_one(negative)

            # Calculate Triplet Loss
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            
            # Backward pass: compute gradients and update model parameters
            loss.backward()
            optimizer.step()

            running_loss += loss.item() # Accumulate batch loss

            # Log training progress periodically
            if (batch_idx + 1) % config['training']['log_interval'] == 0:
                print(f"Epoch [{epoch}/{config['training']['epochs']}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Current Loss: {loss.item():.6f}, "
                      f"Avg Batch Loss: {running_loss / (batch_idx + 1):.6f}")
         
        
        # Calculate average loss for the current epoch
        avg_epoch_loss = running_loss / len(train_loader)
        print(f"\n--- Epoch {epoch} Complete. Average Epoch Loss: {avg_epoch_loss:.6f} ---")


        # --- Test evaluation ---
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Validation Loss after Epoch {epoch}: {val_loss:.6f}")

        # ROC AUC computation removed to lighten training load

        run.log({
            "train_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "epoch": epoch,
            "dataset_size_factor": config['data'].get('dataset_size_factor', 1.0)
        })
        

        # --- 7. Save checkpoint ---
        if val_loss < best_loss:
            best_loss = val_loss
            # Use current timestamp for unique checkpoint name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"model_best_epoch_{epoch}_{timestamp}.pt"
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(checkpoint_dir, checkpoint_name))
            print(f"New best model saved with loss: {best_loss:.6f}")

        # --- Optional: Add validation/evaluation here if you have a separate validation set ---
        # For simplicity, this example focuses on the training loop.
        # A full project would evaluate on a validation set and save the model
        # based on validation loss/metric to prevent overfitting.

    print("\n--- Training Complete ---")
    run.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a Transformer Siamese Network for audio cognate detection.")
    parser.add_argument('--config', type=str, default='experiments/exp01_baseline/config.yaml',
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Create a default config file if it doesn't exist for easy first run
    config_dir = os.path.dirname(args.config)
    os.makedirs(config_dir, exist_ok=True)
    if not os.path.exists(args.config):
        print(f"Creating a default config file at {args.config}")
        default_config = {
            'data': {
                'db_path': 'data/metadata/cognates_2.db',
                'data_base_dir': 'data/processed', # Base directory for mel spectrograms
                'max_frames': 500, # Maximum sequence length for spectrograms
                'dataset_size_factor': 1.0, # Dataset size factor: 1.0 = full dataset, 0.5 = half dataset, 0.25 = quarter dataset
                'use_specaugment': False # Whether to apply SpecAugment during training
            },
            'model': {
                'n_mels': 80,
                'd_model': 256,
                'nhead': 8,
                'num_encoder_layers': 4,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'embedding_dim': 128
            },
            'training': {
                'epochs': 20,
                'batch_size': 32,
                'learning_rate': 0.0001,
                'log_interval': 10, # Log training loss every N batches
                'checkpoint_dir': 'experiments/exp01_baseline/checkpoints',
                'resume_checkpoint': False, # Set to True to resume from the last saved checkpoint
                'resume_checkpoint_name': '', # Specify exact checkpoint file name if resuming (e.g., 'model_best_epoch_X_YYYYMMDD_HHMMSS.pt')
                           # Leave empty if resume_checkpoint is False.
                'num_workers': 4 # Number of data loading workers (set to 0 for debugging on Windows)
            },
            'loss': {
                'margin': 1.0 # Margin for TripletLoss
            }
        }
        with open(args.config, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        print("A default config file has been generated. Please review and adjust it as needed.")
        print("Then run 'python src/training/train.py --config experiments/exp01_baseline/config.yaml' again.")
    else:
        # If config file already exists, proceed with training
        train(args.config)

