import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from classifyMatch import MatchModel
from extent import Extent

# --- Load the model ---
model = MatchModel()
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# --- Set up the transform (must match training preprocessing) ---
transform = transforms.Compose([
    transforms.ToTensor(),
])

# --- Load CSVs ---
grid_df = pd.read_csv("/home/james/Documents/fireHoseSam/mapfiles/gridsearch/gridmetdata.csv")
find_df = pd.read_csv("/home/james/Documents/fireHoseSam/mapfiles/toFindOnGrid/tofindmetadata.csv")

# --- Helper to load and transform an image ---
def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image)


def boxes_overlap(a, b):
    return (
        a["minX"] < b["maxX"] and a["maxX"] > b["minX"] and
        a["minY"] < b["maxY"] and a["maxY"] > b["minY"]
    )


# --- Matching process ---
matches = []  # list of tuples (find_id, grid_id, score) if needed
goodResultCount = 0
badResultCount = 0

for i, find_row in find_df.iterrows():
    currentBestMatchForToFind = None
    find_image_path = find_row["filepath"]
    
    find_img_tensor = load_image(find_image_path).unsqueeze(0)
    findExtent = Extent(find_row["minX"], find_row["maxX"], find_row["minY"], find_row["maxY"])
            
    for _, grid_row in grid_df.iterrows():
        gridExtent = Extent(grid_row["minX"], grid_row["maxX"], grid_row["minY"], grid_row["maxY"])
        shouldOverlap = False
        if findExtent.overlapsOtherExtent(gridExtent):
            shouldOverlap = True


        grid_image_path = grid_row["filepath"]
        grid_img_tensor = load_image(grid_image_path).unsqueeze(0)

        with torch.no_grad():
            output = model(find_img_tensor, grid_img_tensor)
            score = output.item()  # since output is a scalar per match

        threshold = 0.8

        if shouldOverlap and score > threshold:
            goodResultCount += 1
        
        if shouldOverlap and score < threshold:
            badResultCount += 1

        if shouldOverlap == False and score > threshold:
            badResultCount += 1

        if shouldOverlap == False and score < threshold:
            goodResultCount += 1
        
        if score > threshold:
            if currentBestMatchForToFind is None:
                currentBestMatchForToFind = {"matchtile":grid_row["filepath"],"score": score,"overlaps": shouldOverlap}
            else:
                if currentBestMatchForToFind["score"] < score:
                    currentBestMatchForToFind = {"matchtile":grid_row["filepath"],"score": score, "overlaps": shouldOverlap}

    if currentBestMatchForToFind is None:
        print(f"No match for {find_image_path}")
    else:
        print(f"Best match for {find_image_path} overlaps:{currentBestMatchForToFind['overlaps']}")
    if i > 10:
        break

total = goodResultCount + badResultCount

print(f"Good: {goodResultCount} bad: {badResultCount}")
print(f"Good percent: {goodResultCount / total * 100} bad: {badResultCount/ total * 100}")


