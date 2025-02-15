import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

class MapBoundingBoxRegressor(nn.Module):
    def __init__(self):
        super(MapBoundingBoxRegressor, self).__init__()
        # Use ResNet18 as the backbone (pretrained on ImageNet)
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.fc = nn.Identity()  # Remove classification head
        
        # Fully connected layer to predict 4 bounding box coordinates
        self.regressor = nn.Sequential(
            nn.Linear(512 * 1, 256),  # ResNet18 outputs 512 features
            nn.ReLU(),
            nn.Linear(256, 1)  # a single number
        )

    def forward(self, map1, map2):
        # Extract features from both maps
        f1 = self.feature_extractor(map1)
        f2 = self.feature_extractor(map2)
        
        # Concatenate feature vectors
        fused_features = torch.abs(f1 - f2)  # Compute absolute difference
        
        # Predict magicNumber
        magicNumber = self.regressor(fused_features)
        return magicNumber
    


# Initialize model
model = MapBoundingBoxRegressor()
if not torch.cuda.is_available():
    raise Exception('NO GPU')
device = torch.device("cuda")
model.to(device)

# Loss function & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Example Training Loop
def train(model, train_loader, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for map1, map2, true_magic_number in train_loader:
            map1, map2, true_magic_number = map1.to(device), map2.to(device), true_magic_number.to(device)
            
            optimizer.zero_grad()
            predicted_magic_number = model(map1, map2)
            loss = criterion(predicted_magic_number, true_magic_number)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


class MapDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file containing image paths and bounding boxes.
            transform (callable, optional): Optional transform to be applied to images.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load images
        map1_path = self.data.iloc[idx, 0]
        map2_path = self.data.iloc[idx, 1]
        map1 = Image.open(f'mapfiles/output/{map1_path}').convert("RGB")
        map2 = Image.open(f'mapfiles/output/{map2_path}').convert("RGB")

        # Load bounding box
        magic_number = self.data.iloc[idx, 2]
        magic_number = torch.tensor([magic_number], dtype=torch.float32)

        # Apply transforms if specified
        if self.transform:
            map1 = self.transform(map1)
            map2 = self.transform(map2)

        return map1, map2, magic_number

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match ResNet input
    transforms.ToTensor(),          # Convert to tensor
])

# Create dataset and DataLoader
csv_path = "mapfiles/output/metadata.csv"
dataset = MapDataset(csv_file=csv_path, transform=transform)

# Split into training & test sets (80% train, 20% test)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoader for batching
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

train(model, train_loader)

def evaluate(model, test_loader):
    model.eval()  # Set to evaluation mode
    total_mse = 0
    total_mae = 0
    total_accuracy = 0
    count = 0

    with torch.no_grad():  # No gradients needed for evaluation
        for map1, map2, myMagicNumber in test_loader:
            map1, map2, myMagicNumber = map1.to(device), map2.to(device), myMagicNumber.to(device)

            predicted_number = model(map1, map2)

            mse = torch.mean((predicted_number - myMagicNumber) ** 2).item()
            mae = torch.mean(torch.abs(predicted_number - myMagicNumber)).item()
            
            # Calculate accuracy as a percentage
            abs_error = torch.abs(predicted_number - myMagicNumber)
            relative_error = abs_error / (myMagicNumber + 1e-8)  # Avoid division by zero
            batch_accuracy = 100 * (1 - relative_error.clamp(0, 1)).mean().item()

            total_mse += mse * map1.size(0)
            total_mae += mae * map1.size(0)
            total_accuracy += batch_accuracy * map1.size(0)
            count += map1.size(0)

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    avg_accuracy = total_accuracy / count

    print(f"Test MSE: {avg_mse:.6f}")
    print(f"Test MAE: {avg_mae:.6f}")
    print(f"Average Prediction Error: {avg_mae:.6f}")
    print(f"Model Accuracy: {avg_accuracy:.2f}%")

# Run evaluation
evaluate(model, test_loader)
torch.save(model.state_dict(), "model.pth")