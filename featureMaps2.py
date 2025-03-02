import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from myOwnCNN import CustomMapRegressor, CustomMapRegressorMoreLayers, CustomMapRegressorMoreLayers2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomMapRegressorMoreLayers2().to(device)
model.load_state_dict(torch.load("customcnnmodel.pth"))  # Adjust file path
model.eval()  # Set to evaluation mode

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

map1 = Image.open("mapfiles/output/0_shifted.png").convert("RGB")
map2 = Image.open("mapfiles/output/0_starting.png").convert("RGB")

# Apply transformation
map1 = transform(map1).unsqueeze(0).to(device)  # Add batch dimension
map2 = transform(map2).unsqueeze(0).to(device)

def visualize_feature_maps(model, image_tensor):
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        activations = model.conv1(image_tensor)
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i >= activations.shape[1]: break
        ax.imshow(activations[0, i].cpu().numpy(), cmap='viridis')
        ax.axis("off")
    plt.show()

visualize_feature_maps(model, map1)