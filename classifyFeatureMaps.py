import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from classifyMatch import MatchModel
from differentScales import FCNImageLocalization
from myOwnCNN import CustomMapRegressorMoreLayers2

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomMapRegressorMoreLayers2().to(device)
model = MatchModel().to(device)
model.load_state_dict(torch.load("best_model.pth"))  # Adjust file path
# model.load_state_dict(torch.load("customcnnmodel.pth"))  # Adjust file path
model.eval()  # Set to evaluation mode

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((256, 256)),  
    transforms.ToTensor(),
])

# Load example maps
map1 = Image.open("/home/james/Documents/fireHoseSam/mapfiles/output/initial_tile_1.png").convert("RGB")

# Apply transformation
map1 = transform(map1).unsqueeze(0).to(device)  # Add batch dimension

# Function to visualize feature maps
def visualize_feature_maps(cnn_model, image_tensor, stop_at_layer_idx):
    with torch.no_grad():
        x = image_tensor
        for i, layer in enumerate(cnn_model):
            x = layer(x)
            if i == stop_at_layer_idx:
                break

    feature_maps = x.squeeze(0).cpu().numpy()

    # Plot feature maps
    num_filters = feature_maps.shape[0]
    fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
    for i in range(num_filters):
        axes[i].imshow(feature_maps[i], cmap="viridis")
        axes[i].axis("off")
        axes[i].set_title(f"Filter {i+1}")
    plt.show()

# Visualize feature maps from different layers
visualize_feature_maps(model.cnn, map1, 0)  # Visualize after first layer
visualize_feature_maps(model.cnn, map1, 1)  # After second
visualize_feature_maps(model.cnn, map1, 2)  # etc.
visualize_feature_maps(model.cnn, map1, 3)  # etc.
# visualize_feature_maps(model, map1, "conv4")  # Third conv layer