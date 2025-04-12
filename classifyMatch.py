import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from playsound import playsound

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
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # For binary classification
        )



    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        diff = torch.abs(feat1 - feat2)
        out = self.classifier(diff.view(diff.size(0), -1))
        return out.squeeze()

# --- Training ---
def train(model, dataloader, device, epochs=300, patience=10):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    epochs_without_improvement = 0

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

        # --- Early stopping based on training loss ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_without_improvement = 0
            # Optionally save the best model
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

def evaluate(model, dataloader, device):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for tile_img, input_img, label in dataloader:
            tile_img, input_img, label = tile_img.to(device), input_img.to(device), label.to(device)

            outputs = model(tile_img, input_img)
            loss = criterion(outputs, label)
            total_loss += loss.item() * tile_img.size(0)

            preds = (outputs >= 0.5).float()
            
            correct += (preds == label).sum().item()
            total += label.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# --- Main ---
if __name__ == "__main__":
    
    image_dir = "/home/james/Documents/fireHoseSam/mapfiles/output"  # folder containing images
    csv_path = "/home/james/Documents/fireHoseSam/mapfiles/output/metadata.csv"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    dataset = ImagePairDataset(csv_path, image_dir, transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    if not torch.cuda.is_available():
        raise Exception('no GPU')

    device = torch.device("cuda")
    model = MatchModel().to(device)
    model.load_state_dict(torch.load("best_model.pth"))  # Adjust file path
    
    train(model, train_loader, device)
    val_loss, val_acc = evaluate(model, test_loader, device)
    print(f"Eval Loss: {val_loss:.4f} - Eval Accuracy: {val_acc:.4f}")
    # torch.save(model.state_dict(), "classifier.pth")
    playsound('/home/james/Downloads/nokia_brick.mp3')