import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from GridSearchParameterSet import GridSearchParameterSet
from MatchModelSlots import MatchModelSlots
from classifyPrecision import MatchModel, ImagePairDataset

# Paths
csv_file = "/home/james/Documents/fireHoseSam/mapfiles/validation/metadata.csv"
image_dir = "/home/james/Documents/fireHoseSam/mapfiles/validation"
model_path = "best_precision_model.pth"

# Transform (must match training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

# Load dataset
dataset = ImagePairDataset(csv_file, image_dir, transform)
if not torch.cuda.is_available():
    raise Exception("NO CUDA")

device = torch.device("cuda")

combo = {
    'nlayers': 4,
    'downSample': 2,
    'leaky_cnn': True,
    'leaky_classifier': True,
    'base_channels': 32, 
    # 'kernel_size': [3,5],
    'padding': 0,
    'classifier_layers': 4,
    'classifier_hidden': 32,
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



# Load model
model = MatchModelSlots(parameterSet).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Read CSV for image names
df = pd.read_csv(csv_file)

with torch.no_grad():
    for idx in range(len(dataset)):
        tile_img, input_img, label = dataset[idx]
        tile_img = tile_img.unsqueeze(0).to(device)
        input_img = input_img.unsqueeze(0).to(device)
        label_val = int(label.item())

        logits = model(tile_img, input_img)
        prob = torch.sigmoid(logits).item()
        pred = 1 if prob >= 0.5 else 0

        if pred != label_val:
            tile_name = df.iloc[idx]['tileImagePath']
            input_name = df.iloc[idx]['inputImagePath']
            print(f"Misclassified: tile={tile_name}, input={input_name}, label={label_val}, pred={pred}, prob={prob:.3f}")