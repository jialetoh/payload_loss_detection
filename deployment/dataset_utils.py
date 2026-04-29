import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PayloadDataset(Dataset):
    """
    Custom Dataset for Payload Loss Detection.
    Expects a list of tuples containing paths to the initial image, 
    current image, and the label (0 for present, 1 for absent).
    """
    def __init__(self, data_pairs, transform=None):
        self.data_pairs = data_pairs
        self.transform = transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        initial_img_path, current_img_path, label = self.data_pairs[idx]

        # Convert to RGB to ensure 3 channels for pre-trained CNNs
        initial_img = Image.open(initial_img_path).convert('RGB')
        current_img = Image.open(current_img_path).convert('RGB')

        if self.transform:
            # Apply transforms independently to simulate environmental changes
            initial_img = self.transform(initial_img)
            current_img = self.transform(current_img)

        label_tensor = torch.tensor([label], dtype=torch.float32)

        return initial_img, current_img, label_tensor

def get_transforms(is_train=True, use_augmentation=True, resolution=(144, 256)):
    """
    Generates the transformation pipeline.
    """
    transform_list = [
        transforms.Resize(resolution)
    ]

    if is_train and use_augmentation:
        transform_list.extend([
            # Simulates chassis vibrations
            transforms.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            # Simulates overhead hangar lighting changes
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5) 
        ])

    transform_list.extend([
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transforms.Compose(transform_list)

def denormalize(tensor):
    """Reverts ImageNet normalization for visualization."""
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    return inv_normalize(tensor)
    