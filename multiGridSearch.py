import json
import os
import torch
import pandas as pd
from pathlib import Path

from shapely.geometry import Polygon, mapping
from PIL import Image
from torchvision import transforms


from GridSearchParameterSet import GridSearchParameterSet
from MatchModelSlots import MatchModelSlots


# Define the transform (grayscale + tensor)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])



if not torch.cuda.is_available():
    raise Exception('NO GPU')


combo = {
    'nlayers': 4,
    'downSample': None,
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

model = MatchModelSlots(parameterSet)
model.load_state_dict(torch.load("best_precision_model.pth"))
model.eval()

def classify_image_pair(img_path1, img_path2, model):
    # Load and preprocess images
    img1 = Image.open(img_path1).convert('RGB')
    img2 = Image.open(img_path2).convert('RGB')
    img1 = transform(img1).unsqueeze(0)  # shape: [1, 1, 128, 128]
    img2 = transform(img2).unsqueeze(0)

    # Move to GPU if available
    device = torch.device("cuda")
    img1 = img1.to(device)
    img2 = img2.to(device)
    model = model.to(device)

    # Inference
    with torch.no_grad():
        output = model(img1, img2)
        prob = torch.sigmoid(output).item()

    return prob

def row_to_polygon(row):
    # Create a rectangle polygon from minX, maxX, minY, maxY
    return Polygon([
        (row['minX'], row['minY']),
        (row['minX'], row['maxY']),
        (row['maxX'], row['maxY']),
        (row['maxX'], row['minY']),
        (row['minX'], row['minY'])
    ])



folder_a = "/home/james/Documents/fireHoseSam/mapfiles/gridsearch/"
folder_b = "/home/james/Documents/fireHoseSam/mapfiles/subgrid/"


gridMetadata = pd.read_csv(Path(folder_a,'gridmetdata.csv'))
subGridMetadata = pd.read_csv(Path(folder_b, 'tofindmetadata.csv'))

def indices_to_dictionary_key(x,y):
    return f"{x}_{y}"

features = {}
for _, row in gridMetadata.iterrows():
    poly = row_to_polygon(row)
    feature = {
        "type": "Feature",
        "geometry": mapping(poly),
        "properties": {
            "x": row['x'],
            "y": row['y'],
            "filepath": row['filepath'],
            "matchcount": 0
        }
    }
    features[indices_to_dictionary_key(row['x'],row['y'])] = feature

pngs_a = [os.path.join(folder_a, f) for f in os.listdir(folder_a) if f.endswith(".png")]
pngs_b = [os.path.join(folder_b, f) for f in os.listdir(folder_b) if f.endswith(".png")]

total = len(gridMetadata) * len(subGridMetadata)
iterCount = 0
results = []

sub_grid_min_x = 11
sub_grid_min_y = 6

sub_grid_max_x = 15
sub_grid_max_y = 17

sub_match_dict = {}

for _, row_a in gridMetadata.iterrows():
    for _, row_b in subGridMetadata.iterrows():
        iterCount += 1
        if iterCount % 1000 == 0:
            print(f"{iterCount/total * 100}% complete")
        if row_b['x'] < sub_grid_min_x or row_b['x'] > sub_grid_max_x:
            continue
        if row_b['y'] < sub_grid_min_y or row_b['y'] > sub_grid_max_y:
            continue

        
        img_a = row_a['filepath']
        img_b = row_b['filepath']
        prob = classify_image_pair(img_a, img_b, model)
        if (prob > 0.7):
            features[indices_to_dictionary_key(row_a['x'],row_a['y'])]['properties']['matchcount'] += 1
            subGridKey = indices_to_dictionary_key(row_b['x'], row_b['y'])
            if subGridKey in sub_match_dict:
                sub_match_dict[subGridKey] += 1
            else:
                sub_match_dict[subGridKey] = 1


featureList = []

for key in features:
    featureList.append(features[key])

geojson_dict = {
    "type": "FeatureCollection",
    "features": featureList
}

with open("grid_polygons.geojson", "w") as f:
    json.dump(geojson_dict, f)

with open('sub_grid_match.json','w') as f:
    json.dump(sub_match_dict, f, indent=4)

# print(classify_image_pair(
#     "/home/james/Documents/fireHoseSam/mapfiles/gridsearch/grid_8_6.png",
#     "/home/james/Documents/fireHoseSam/mapfiles/subgrid/grid_0_0.png",
#     model
# ))