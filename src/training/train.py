"""
Training Script - Debug Version
"""

import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from pathlib import Path

import sys
sys.path.append('.')

from src.models.unet import SegmentationModel
from src.data.dataset import SegmentationDataset
from src.data.augmentation import get_train_aug, get_valid_aug


def train_fn(data_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0

    for images, masks in tqdm(data_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits, loss = model(images, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(data_loader):
            images = images.to(device)
            masks = masks.to(device)
            logits, loss = model(images, masks)
            total_loss += loss.item()
            
    return total_loss / len(data_loader)


def train():
    # Load config with debug
    print("Loading config from configs/config.yaml...")
    try:
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        print(f"Config loaded successfully!")
        print(f"Config keys: {config.keys() if config else 'None'}")
    except Exception as e:
        print(f"ERROR loading config: {e}")
        raise
    
    if config is None:
        print("ERROR: Config is None after loading!")
        return
    
    # Settings from config
    csv_file = config['data']['csv_file']
    epochs = config['training']['epochs']
    LR = config['training']['learning_rate']
    image_size = config['data']['image_size']
    encoder = config['model']['encoder']
    weights = config['model']['encoder_weights']
    batch_size = config['training']['batch_size']
    
    # Auto-detect device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_file)
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    
    print(f"Train samples: {len(train_df)}, Valid samples: {len(valid_df)}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SegmentationDataset(train_df, get_train_aug(image_size))
    valid_dataset = SegmentationDataset(valid_df, get_valid_aug(image_size))
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create model
    print("Creating model...")
    model = SegmentationModel(encoder=encoder, weights=weights)
    model = model.to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # Create checkpoint directory
    Path('checkpoints').mkdir(exist_ok=True)
    
    # Training loop
    best_valid_loss = float('inf')
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        train_loss = train_fn(train_loader, model, optimizer, DEVICE)
        valid_loss = eval_fn(valid_loader, model, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Valid Loss: {valid_loss:.4f}")
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Saved best model!")
    
    print("\nTraining completed!")


if __name__ == '__main__':
    train()
