"""
Quick test to verify setup is working
"""

print("Testing imports...")

try:
    import torch
    print("PyTorch imported")
    
    import cv2
    print("OpenCV imported")
    
    import pandas as pd
    print("Pandas imported")
    
    import albumentations
    print("Albumentations imported")
    
    import segmentation_models_pytorch as smp
    print("Segmentation Models PyTorch imported")
    
    from src.models.unet import SegmentationModel
    print("Your UNet model imported")
    
    from src.data.dataset import SegmentationDataset
    print("Your Dataset imported")
    
    from src.data.augmentation import get_train_aug, get_valid_aug
    print("Your Augmentation functions imported")
    
    print("\nAll imports successful! Your setup is working correctly.")
    
except Exception as e:
    print(f"\nError: {e}")
