import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# Custom CNN for Map Feature Extraction
class CustomMapRegressor(nn.Module):
    def __init__(self):
        super(CustomMapRegressor, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * (224 // 8) * (224 // 8), 256)  # Adjusted for 224x224 input size
        self.fc2 = nn.Linear(256, 1)  # Single output: magic number

    def forward(self, map1, map2):
        # Feature extraction for both maps
        f1 = self.pool(torch.relu(self.conv1(map1)))
        f1 = self.pool(torch.relu(self.conv2(f1)))
        f1 = self.pool(torch.relu(self.conv3(f1)))

        f2 = self.pool(torch.relu(self.conv1(map2)))
        f2 = self.pool(torch.relu(self.conv2(f2)))
        f2 = self.pool(torch.relu(self.conv3(f2)))

        # Flatten
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)

        # Compute absolute difference
        fused_features = torch.abs(f1 - f2)

        # Fully connected layers
        x = torch.relu(self.fc1(fused_features))
        magicNumber = self.fc2(x)
        
        return magicNumber



class CustomMapRegressorMoreLayers(nn.Module):
    def __init__(self):
        super(CustomMapRegressorMoreLayers, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # NEW LAYER
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * (224 // 16) * (224 // 16), 256)  
        self.fc2 = nn.Linear(256, 1)

    def forward(self, map1, map2):
        f1 = self.pool(torch.relu(self.bn1(self.conv1(map1))))
        f1 = self.pool(torch.relu(self.bn2(self.conv2(f1))))
        f1 = self.pool(torch.relu(self.bn3(self.conv3(f1))))
        f1 = self.pool(torch.relu(self.bn4(self.conv4(f1))))

        f2 = self.pool(torch.relu(self.bn1(self.conv1(map2))))
        f2 = self.pool(torch.relu(self.bn2(self.conv2(f2))))
        f2 = self.pool(torch.relu(self.bn3(self.conv3(f2))))
        f2 = self.pool(torch.relu(self.bn4(self.conv4(f2))))

        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)

        fused_features = torch.abs(f1 - f2)

        x = torch.relu(self.fc1(fused_features))
        magicNumber = self.fc2(x)
        
        return magicNumber


class CustomMapRegressorMoreLayers2(nn.Module):
    def __init__(self):
        super(CustomMapRegressorMoreLayers2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        
        # Compute final feature map size after convolutions (assuming 224x224 input)

        final_size = 224 // (2**4)  # Each stride-2 conv reduces size by half
        feature_size = 256 * final_size * final_size  # Original flattened size
        self.fc1 = nn.Linear(3 * feature_size, 256)  # Multiply by 3 to match new fused_features size


        # final_size = 224 // (2**4)  # Each stride-2 conv reduces size by half
        # self.fc1 = nn.Linear(256 * final_size * final_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, map1, map2):
        f1 = torch.relu(self.bn1(self.conv1(map1)))
        f1 = torch.relu(self.bn2(self.conv2(f1)))
        f1 = torch.relu(self.bn3(self.conv3(f1)))
        f1 = torch.relu(self.bn4(self.conv4(f1)))

        f2 = torch.relu(self.bn1(self.conv1(map2)))
        f2 = torch.relu(self.bn2(self.conv2(f2)))
        f2 = torch.relu(self.bn3(self.conv3(f2)))
        f2 = torch.relu(self.bn4(self.conv4(f2)))

        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)

        fused_features = torch.abs(f1 - f2)
        fused_features = torch.cat([f1, f2, torch.abs(f1 - f2)], dim=1)
        
        
        x = torch.relu(self.fc1(fused_features))
        magicNumber = self.fc2(x)
        
        return magicNumber



# Initialize model
model = CustomMapRegressorMoreLayers2()
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

if __name__ == '__main__':


    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to match CNN input
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
    print('training starting')
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
    print('Accuracy of test data is: ')
    evaluate(model, test_loader)
    print('Accuracy of training data is: ')
    evaluate(model, train_loader)

    torch.save(model.state_dict(), "customcnnmodel.pth")