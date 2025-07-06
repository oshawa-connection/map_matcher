'''
Before moving to grid search.
'''
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

class ImagePairDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, row_limit=None):
        self.data = pd.read_csv(csv_file)
        if (row_limit is not None):
            self.data = self.data.head(row_limit)
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

class MatchModel(nn.Module):
    def __init__(self):
        super(MatchModel, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
            # No sigmoid — using BCEWithLogitsLoss
        )

    def forward(self, img1, img2):
        feat1 = self.cnn(img1)
        feat2 = self.cnn(img2)
        diff = torch.abs(feat1 - feat2)
        out = self.classifier(diff.view(diff.size(0), -1))
        return out  # shape [B, 1]


class GridSearchParameterSet:
    def __init__(nLayers: int, dropout: bool, flatten: bool,padding:bool, relu: bool, downSample:bool, leaky: bool):
        pass




# class MatchModelSlots(nn.Module):
#     def __init__(self,  parameterSet: GridSearchParameterSet):
#         super(MatchModel, self).__init__()
    

#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, kernel_size=3),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((4, 4))
#         )

#         self.classifier = nn.Sequential(
#             nn.Linear(1024, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#             # No sigmoid — using BCEWithLogitsLoss
#         )

#     def forward(self, img1, img2):
#         feat1 = self.cnn(img1)
#         feat2 = self.cnn(img2)
#         diff = torch.abs(feat1 - feat2)
#         out = self.classifier(diff.view(diff.size(0), -1))
#         return out  # shape [B, 1]


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()  # use same loss as training

    true_positives = 0
    predicted_positives = 0

    with torch.no_grad():
        for tile_img, input_img, label in dataloader:
            total += label.size(0)
            tile_img = tile_img.to(device)
            input_img = input_img.to(device)
            label = label.to(device).unsqueeze(1)  # shape [B, 1]

            logits = model(tile_img, input_img)
            loss = criterion(logits, label)
            total_loss += loss.item() * tile_img.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == label).sum().item()

            true_positives += ((preds == 1) & (label == 1)).sum().item()
            predicted_positives += (preds == 1).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total
    precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
    return avg_loss, accuracy, precision

def train(model, dataloader, val_loader, device, epochs=50, patience=10):
    model = model.to(device)

    pos_weight = torch.tensor([10.0], device=device)  # Heavily penalize false positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = -1
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for tile_img, input_img, label in dataloader:
            tile_img = tile_img.to(device)
            input_img = input_img.to(device)
            label = label.to(device).unsqueeze(1)  # shape [B, 1]

            optimizer.zero_grad()
            logits = model(tile_img, input_img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * tile_img.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

        val_loss, val_accuracy, val_precision = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.4f} | Precision: {val_precision:.4f}")

        if val_precision > best_val_loss:
            best_val_loss = val_precision
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_precision_model.pth")
            print("Saved best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break


def basicTraining():
        # image_dir = "/home/james/Documents/fireHoseSam/mapfiles/output"
    # csv_path = "/home/james/Documents/fireHoseSam/mapfiles/output/metadata.csv"

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), # 3 channel PNG is black and white
        transforms.ToTensor()
    ])

    train_dataset = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/output/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/output", transform, 10_000)
    test_dataset = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/validation/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/validation", transform, 5_000)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    if not torch.cuda.is_available():
        raise Exception('No GPU available')

    device = torch.device("cuda")
    model = MatchModel().to(device)

    print('Starting training...')
    train(model, train_loader, test_loader, device, 100, 100)

# --- Main ---
if __name__ == "__main__":
    basicTraining()