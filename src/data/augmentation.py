"""
Data Augmentation Functions - Your Original Code
"""

import albumentations as a


def get_train_aug(image_size=320):
    return a.Compose([
        a.Resize(image_size, image_size),
        a.HorizontalFlip(p=0.5),
        a.VerticalFlip(p=0.5)
    ], is_check_shapes=False)


def get_valid_aug(image_size=320):
    return a.Compose([
        a.Resize(image_size, image_size)
    ], is_check_shapes=False)
