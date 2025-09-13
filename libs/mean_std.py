import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch
from tqdm import tqdm

class ImageFolderDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = []

        # Duyệt tất cả subfolders để lấy ảnh
        for root, _, files in os.walk(root_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(os.path.join(root, f))

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    n_images = 0
    mean = 0.
    std = 0.

    for images in tqdm(loader, desc="Tính mean/std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        n_images += batch_samples

    mean /= n_images
    std /= n_images
    return mean, std

if __name__ == "__main__":
    dataset_path = "../data/train"
    dataset = ImageFolderDataset(dataset_path)

    mean, std = compute_mean_std(dataset)
    print(f"Mean: {mean}")
    print(f"Std: {std}")
