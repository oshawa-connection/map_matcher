'''
Before moving to grid search.
'''
import torch.optim.lr_scheduler
from pynput import keyboard
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms
from itertools import product

from GridSearchParameterSet import GridSearchParameterSet
from MatchModelSlots import MatchModelSlots


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




# ...existing code...

def trainGrid(learning_rate, model, dataloader, val_loader, device, epochs=100, patience=10, patience_delta = 0.0001) -> float:
    model = model.to(device)

    pos_weight = torch.tensor([10.0], device=device)  # Heavily penalize false positives
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Add a scheduler that reduces LR when validation precision plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )

    best_val_loss = -1.0
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

        # Step the scheduler with the validation precision
        scheduler.step(val_precision)

        if val_precision > best_val_loss:
            if (val_precision - best_val_loss) > patience_delta:            
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            best_val_loss = val_precision

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                return best_val_loss
            
    return best_val_loss

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), # 3 channel PNG is black and white
    transforms.ToTensor()
])


train_dataset_small = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/output/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/output", transform, 5_000)
test_dataset_small = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/validation/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/validation", transform, 2_500)

train_loader_small = DataLoader(train_dataset_small, batch_size=64, shuffle=True)
test_loader_small = DataLoader(test_dataset_small, batch_size=64, shuffle=True)


train_dataset_big = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/output/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/output", transform)
test_dataset_big = ImagePairDataset("/home/james/Documents/fireHoseSam/mapfiles/validation/metadata.csv", "/home/james/Documents/fireHoseSam/mapfiles/validation", transform)

train_loader_big = DataLoader(train_dataset_big, batch_size=32, shuffle=True)
test_loader_big = DataLoader(test_dataset_big, batch_size=32, shuffle=True)


if not torch.cuda.is_available():
    raise Exception('No GPU available')

device = torch.device("cuda")


def basicTraining():

    model = MatchModel().to(device)

    print('Starting training...')
    train(model, train_loader_small, test_loader_small, device, 100, 100)

stop_switch = False

def on_press(key):
    global stop_switch
    try:
        if key.char == 'q':
            stop_switch = True
            print('----------------- q key pressed, ending after this iteration has finished ----------------- ')
            # Returning False would stop the listener, but we want to finish the iteration
    except AttributeError:
        pass  # Special keys (ctrl, etc.) are ignored


def gridSearch():

    with open('gridSearch.dicts','r') as existing_dict_file:
        num_lines = sum(1 for _ in existing_dict_file)

    print(f'skipping {num_lines} combinations')
    with open('out.txt', 'a') as f, open('gridSearch.dicts','a') as dict_file, keyboard.Listener(on_press=on_press) as listener:
        # (self, nLayers: int, dropout: bool = False, flatten: bool = False, downSample:bool = False, leaky: bool = False, base_channels = 16,kernel_size =3,padding = 0):
        search_space2 = {
            'nlayers': [3, 4],
            'downSample': [2], #[None, 2]
            'leaky_cnn': [True, False],
            'leaky_classifier': [True, False],
            'base_channels': [16, 32],
            # 'kernel_size': [3,5],
            'padding': [0],
            'classifier_layers': [2, 3],
            'classifier_hidden': [64, 128],
            # NEW
            # 'dropout': [0.0, 0.2],
            # 'learning_rate': [1e-4, 1e-3],
        }


        search_space = {
            'nlayers': [5], # change this to 6 later
            'downSample': [2],
            'leaky_cnn': [True],
            'leaky_classifier': [False],
            'base_channels': [32], 
            # 'kernel_size': [3,5],
            'padding': [0],
            'classifier_layers': [4],
            'classifier_hidden': [128],
            'learning_rate': [1e-2,1e-3,1e-4,1e-5]
        }

        # Get keys and values
        keys = search_space.keys()
        values = search_space.values()

        # Create list of all combinations
        combinations = [dict(zip(keys, v)) for v in product(*values)]
        # def __init__(self, nLayers: int, dropout: bool = False, flatten: bool = False, downSample:bool = False, leaky: bool = False, base_channels = 16,kernel_size =3,padding = 0):
        
        for i in range(num_lines ,len(combinations)):
            if stop_switch:
                print('Goodbye')
                break
            combo = combinations[i]
            parameterSet = GridSearchParameterSet(
                nLayers = combo['nlayers'], 
                downSample=combo['downSample'],
                leaky_cnn=combo['leaky_cnn'],
                leaky_classifier=combo['leaky_classifier'],
                base_channels=combo['base_channels'],
                # kernel_size= combo['kernel_size'],
                padding= combo['padding'],
                classifier_layers=combo['classifier_layers'],
                classifier_hidden=combo['classifier_hidden']
            )
            
            print(f"Starting CNN params: {parameterSet.feature_maps}, classifier params: {parameterSet.classifier_params}")
            print(combo)
            print('\n')
            model = MatchModelSlots(parameterSet).to(device)
            try:
                best_precision = trainGrid(combo['learning_rate'],model, train_loader_small, test_loader_small, device, 20, 10)
                print(f"finished; best precision was: {best_precision}")
                f.write(f"{combo},{best_precision}\n")
                dict_file.write(f"{combo}\n")
                dict_file.flush()
                f.flush()
            except Exception as e:
                print(e)
                print(f"FAILED")
                f.write(f"{combo},FAILED\n")
                dict_file.write(f"{combo}\n")
                dict_file.flush()
                f.flush()

def train_with_early_quit(learning_rate, model, dataloader, val_loader, device, epochs=100, patience=10, patience_delta=0.0001) -> float:
    pos_weight = torch.tensor([10.0], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True, min_lr=1e-6
    )

    best_val_loss = -1.0
    epochs_without_improvement = 0

    with keyboard.Listener(on_press=on_press) as _:
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for tile_img, input_img, label in dataloader:
                tile_img = tile_img.to(device)
                input_img = input_img.to(device)
                label = label.to(device).unsqueeze(1)

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

            scheduler.step(val_precision)

            should_save = epoch > 10

            if val_precision > best_val_loss:
                if (val_precision - best_val_loss) > patience_delta:
                    epochs_without_improvement = 0
                    if (should_save):
                        print('saving best model')
                        torch.save(model.state_dict(), "best_precision_model.pth")
                else:
                    epochs_without_improvement += 1
                best_val_loss = val_precision
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

            if stop_switch:
                # Save model and quit after finishing this epoch
                save_path = f"early_finish_epoch_{epoch+1}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Early quit requested. Model saved to {save_path}")
                break

    return best_val_loss
   
def big_refine():

    combo = {
        'nlayers': 4,
        'downSample': 2,
        'leaky_cnn': True,
        'leaky_classifier': False,
        'base_channels': 32, 
        # 'kernel_size': [3,5],
        'padding': 0,
        'classifier_layers': 4,
        'classifier_hidden': 128,
        'learning_rate': 1e-3
    }

    parameterSet = GridSearchParameterSet(
        nLayers = combo['nlayers'], 
        downSample=combo['downSample'],
        leaky_cnn=combo['leaky_cnn'],
        leaky_classifier=combo['leaky_classifier'],
        base_channels=combo['base_channels'],
        # kernel_size= combo['kernel_size'],
        padding= combo['padding'],
        classifier_layers=combo['classifier_layers'],
        classifier_hidden=combo['classifier_hidden']
    )

    model = MatchModelSlots(parameterSet).to(device)
    checkpoint_path = "early_finish_epoch_26.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

   
    print('Starting training...')
    train_with_early_quit(combo['learning_rate'], model, train_loader_big, test_loader_big, device, 300, 15)


# --- Main ---
if __name__ == "__main__":
    # basicTraining()
    # gridSearch()
    big_refine()