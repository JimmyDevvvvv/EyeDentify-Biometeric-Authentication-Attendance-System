# train_model.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model.iris_net import IrisNet
from utils.preprocess import preprocess_image

# =========================
# 🔧 Settings
# =========================
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "model/iris_model.pth"
BATCH_SIZE = 8
EPOCHS = 15
LEARNING_RATE = 0.001

# =========================
# 🧾 Custom Dataset Loader
# =========================
class IrisDataset(Dataset):
    def __init__(self, dataset_path):
        self.data = []
        self.labels = []
        self.label_map = {}
        self.transform = transforms.ToTensor()

        users = sorted(os.listdir(dataset_path))
        for idx, user in enumerate(users):
            self.label_map[idx] = user
            user_folder = os.path.join(dataset_path, user)
            for img_name in os.listdir(user_folder):
                img_path = os.path.join(user_folder, img_name)
                self.data.append(img_path)
                self.labels.append(idx)

        self.num_classes = len(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100)) / 255.0
        img = np.expand_dims(img, axis=0)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        label = self.labels[idx]
        return img_tensor, label

# =========================
# 🏋️ Train Model
# =========================
def train():
    dataset = IrisDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = IrisNet(num_classes=dataset.num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"[INFO] Starting training on {dataset.num_classes} users...")

    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total * 100
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {total_loss:.4f} | Accuracy: {accuracy:.2f}%")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\n✅ Model trained and saved to {MODEL_SAVE_PATH}")
    print(f"🎯 Classes learned: {dataset.label_map}")

if __name__ == "__main__":
    train()
