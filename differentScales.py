import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os

if not torch.cuda.is_available():
    raise Exception('NO GPU')

# Check for GPU availability
device = torch.device("cuda")
print(f"Using device: {device}")

# Custom Dataset for Large and Small Image Pairs
class ImageLocalizationDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        large_image = Image.open(f"mapfiles/output/{row['starting']}").convert("RGB")
        small_image = Image.open(f"mapfiles/output/{row['shifted']}").convert("RGB")
        
        bbox = torch.tensor([row['minX'], row['minY'], row['maxX'], row['maxY']], dtype=torch.float32)
        
        if self.transform:
            large_image = self.transform(large_image)
            small_image = self.transform(small_image)
        
        return large_image, small_image, bbox

# Fully Convolutional Network (FCN) Model
class FCNImageLocalization(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Simple CNN-based feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 1x1 Conv to generate a heatmap
        self.heatmap_head = nn.Conv2d(256, 1, kernel_size=1)
        
    def forward(self, large_image):
        features = self.encoder(large_image)  # Extract features
        heatmap = self.heatmap_head(features)  # Generate heatmap
        heatmap = F.interpolate(heatmap, size=large_image.shape[2:], mode='bilinear', align_corners=False)
        return heatmap

# Loss function (Mean Squared Error between predicted heatmap and ground truth)
def heatmap_loss(predicted_heatmap, bbox, image_size):
    batch_size, _, h, w = predicted_heatmap.shape
    target_heatmap = torch.zeros((batch_size, 1, h, w), device=predicted_heatmap.device)

    for i in range(batch_size):
        x_min, y_min, x_max, y_max = bbox[i]
        x_min, x_max = int(x_min * w / image_size[1]), int(x_max * w / image_size[1])
        y_min, y_max = int(y_min * h / image_size[0]), int(y_max * h / image_size[0])
        target_heatmap[i, 0, y_min:y_max, x_min:x_max] = 1.0  # Mark region with 1

    return F.mse_loss(predicted_heatmap, target_heatmap)

# Example Usage
if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    # Load dataset
    dataset = ImageLocalizationDataset("mapfiles/output/metadata.csv", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Initialize model and move it to GPU if available
    model = FCNImageLocalization().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(10):
        for large_image, small_image, bbox in dataloader:
            # Move data to GPU
            large_image = large_image.to(device)
            small_image = small_image.to(device)
            bbox = bbox.to(device)

            optimizer.zero_grad()
            heatmap = model(large_image)
            loss = heatmap_loss(heatmap, bbox, (512, 512))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "differentscales.pth")