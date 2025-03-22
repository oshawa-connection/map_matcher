import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

if not torch.cuda.is_available():
    raise Exception('NO GPU')

# --- Dataset Class ---
class ImagePairDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tile_path = os.path.join(self.image_dir, self.data.iloc[idx]['tileImagePath'])
        input_path = os.path.join(self.image_dir, self.data.iloc[idx]['inputImagePath'])
        label = float(self.data.iloc[idx]['doTheyMatch'])

        tile_image = Image.open(tile_path).convert('RGB')
        input_image = Image.open(input_path).convert('RGB')

        if self.transform:
            tile_image = self.transform(tile_image)
            input_image = self.transform(input_image)

        return tile_image, input_image, torch.tensor(label, dtype=torch.float32)

# --- Model ---
class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()

        # Shared CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Input: [B, 3, H, W]
            nn.ReLU(),
            nn.MaxPool2d(2),  # Downsample

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Compress to fixed size
        )

        # Classifier on top of concatenated features
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4 * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # For binary classification
        )

    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        concat = torch.cat((feat1.view(feat1.size(0), -1),
                            feat2.view(feat2.size(0), -1)), dim=1)
        out = self.classifier(concat)
        return out.squeeze()

# --- Training ---
def train(model, dataloader, device, epochs=10):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for tile_img, input_img, label in dataloader:
            tile_img, input_img, label = tile_img.to(device), input_img.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(tile_img, input_img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tile_img.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# --- Main ---
if __name__ == "__main__":
    image_dir = "./images"  # folder containing images
    csv_path = "./train_data.csv"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    dataset = ImagePairDataset(csv_path, image_dir, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = MatchModel()
    train(model, dataloader, device=torch.device("cuda"))