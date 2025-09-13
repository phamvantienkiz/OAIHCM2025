### START: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###
SEED = 25 # Số seed của BTC

# Đường dẫn đến thư mục train và test
# Phần này cấu trúc giống mẫu của BTC
TRAIN_DATA_DIR_PATH = "./data/train"
TEST_DATA_DIR_PATH = "./hutech_oai_2025_test" # Thay đổi đường dẫn thư mục test của BTC ở đây ạ!

# Các khai báo riêng
VAL_SPLIT = 0.2

# Phần này chỉ tính toán trong tập train,
# BTC có thể kiểm tra trong "libs/mean_std"
MEAN = [0.4397, 0.3948, 0.3603]
STD = [0.1841, 0.1777, 0.1702]

# Phần này là đường dẫn thư mục lưu model và output
# BTC thay đường dẫn hoặc chỉnh sửa phù hợp giúp em ạ
MODEL_PATH = "models_saved/best.pt"
OUTPUT_CSV = "output/results.csv"

CATEGORIES = ["NM", "BN", "DG", "LC"]  # Class 0,1,2,3
IMAGE_SIZE = (32, 32)

### END: CÁC KHAI BÁO CHÍNH - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN IMPORT ###
# Các thư viện và phiên bản đơợc khai báo trong requirements.txt
import torch.cuda
import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomRotation, ColorJitter
import torch.nn as nn
from torch.optim import Adam

from sklearn.metrics import accuracy_score
from tqdm.autonotebook import tqdm
import os
import pandas as pd
from PIL import Image
### END: CÁC THƯ VIỆN IMPORT ###

### START: SEEDING EVERY THING - KHÔNG THAY ĐỔI ###
# Set seed for numpy
np.random.seed(SEED)
# Set seed for torch
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
### END: SEEDING EVERY THING - KHÔNG THAY ĐỔI ###

### START: CÁC THƯ VIỆN DATASET, MODEL riêng của nhóm ###
from libs.dataset import OaiDataset
from libs.model import MyResNet50
### END: CÁC THƯ VIỆN DATASET, MODEL riêng của nhóm ###


### START: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###
def train(batch_size, learning_rate, resume_training, num_epoch):
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cho train
    train_transform = Compose([
        Resize(IMAGE_SIZE),
        RandomHorizontalFlip(p=0.2),
        RandomRotation(degrees=14),
        ColorJitter(brightness=0.2, contrast=0.1),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])

    # Cho val/test
    val_transform = Compose([
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])

    train_dataset = OaiDataset(train_dir=TRAIN_DATA_DIR_PATH, mode="train", transform=train_transform, val_split=VAL_SPLIT, seed=SEED)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    val_dataset = OaiDataset(train_dir=TRAIN_DATA_DIR_PATH, mode="val", transform=val_transform, val_split=VAL_SPLIT, seed=SEED)
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False
    )

    # Load model
    model = MyResNet50(num_classes=4)
    model.to(device)

    # Loss func
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = Adam(params=model.parameters(), lr=learning_rate, weight_decay=1e-4)

    if resume_training:
        checkpoint = torch.load("models_saved/last.pt")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_acc = checkpoint["accuracy"]
    else:
        start_epoch = 0
        best_acc = -1

    # Train
    no_improve_epochs = 0
    for epoch in range(start_epoch, num_epoch):
        # Training mode
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        for iter, (images, labels) in enumerate(progress_bar):
            # To GPU
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            output = model(images)
            loss_value = criterion(output, labels) #(input, target)
            progress_bar.set_description("Epoch {}/{}. Loss {:0.4f}".format(epoch + 1, num_epoch, loss_value.item()))
            # Backward
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

        # Evaluation mode
        model.eval()
        losses = []
        all_predictions = []
        all_labels = []
        progress_bar = tqdm(val_dataloader, colour="green")
        for iter, (images, labels) in enumerate(progress_bar):
            # To GPU
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            with torch.inference_mode():
                output = model(images)
            loss_value = criterion(output, labels) #(input, target)
            losses.append(loss_value.item())
            predictions = torch.argmax(output, dim=1)
            all_predictions.extend(predictions.tolist()) # type = tensor
            all_labels.extend(labels.tolist())
        acc = accuracy_score(all_labels, all_predictions)
        loss = np.mean(losses)
        print("Epoch {}/{}.  Loss {:0.4f} Accuracy {:0.4f}".format(epoch + 1, num_epoch, loss, acc))

        checkpoint = {
            "epoch": epoch+1,
            "accuracy": best_acc,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        torch.save(checkpoint, "models_saved/last.pt")
        early_stopping_patience = 5
        if acc > best_acc:
            best_acc = acc
            torch.save(checkpoint, "models_saved/best.pt")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

if __name__ == '__main__':
    train_dir = "data/train"
    test_dir = "data/test"
    batch_size = 32
    learning_rate = 0.0001
    num_epoch = 20
    val_split = 0.2
    resume_training = False

    train(batch_size, learning_rate, resume_training, num_epoch)
### END: ĐỊNH NGHĨA & CHẠY HUẤN LUYỆN MÔ HÌNH ###

# Phần của nhóm em khi quá trình training kết thúc sẽ chạy ngay phần thực nghiệm và xuất file

### START: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###

    # ==== Transform ====
    transform = Compose([
        Resize(IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=MEAN, std=STD)
    ])

    # ==== Load model ====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyResNet50(num_classes=4)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    # ==== Inference ====
    results = []

    for filename in sorted(os.listdir(TEST_DATA_DIR_PATH)):
        if filename.endswith(".jpg"):
            img_path = os.path.join(TEST_DATA_DIR_PATH, filename)
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dim

            with torch.no_grad():
                output = model(image_tensor)
                pred = torch.argmax(output, dim=1).item()

            img_id = os.path.splitext(filename)[0]  # Lấy phần '001' từ '001.jpg'
            results.append({
                "id": img_id,
                "type": pred
            })

    # ==== Save CSV ====
    df = pd.DataFrame(results)
    df = df.sort_values(by="id")  # Đảm bảo thứ tự nếu cần
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Đã lưu kết quả vào {OUTPUT_CSV}")

### END: THỰC NGHIỆM & XUẤT FILE KẾT QUẢ RA CSV ###