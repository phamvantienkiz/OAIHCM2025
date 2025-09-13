from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
import os
import random
from PIL import Image

class OaiDataset(Dataset):
    def __init__(self, train_dir, mode, transform=None, val_split=0.2, seed=42):
        self.categories = ["NM", "BN", "DG", "LC"]
        random.seed(seed)

        self.image_paths = []
        self.labels = []
        for index, category in enumerate(self.categories):
            category_path = os.path.join(train_dir, category)
            all_image_paths = []
            for name in os.listdir(category_path):
                if name.endswith(".jpg"):
                    full_path = os.path.join(category_path, name)
                    all_image_paths.append(full_path)

            random.shuffle(all_image_paths)
            split_index = int(len(all_image_paths) * (1 - val_split))
            if mode == "train":
                chosen = all_image_paths[:split_index]
            elif mode == "val":
                chosen = all_image_paths[split_index:]

            self.image_paths.extend(chosen)
            self.labels.extend([index for _ in chosen])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[index]

        return image, label


if __name__ == '__main__':
    train_dir = "../data/train"
    mode = "train"
    val_split = 0.2
    seed = 42
    transform = ToTensor()
    dataset = OaiDataset(train_dir,"train", transform, val_split, seed)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )
    for images, labels in dataloader:
        print(images.shape, labels)