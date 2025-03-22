import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from differentScales import FCNImageLocalization
from myOwnCNN import CustomMapRegressorMoreLayers2

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CustomMapRegressorMoreLayers2().to(device)
model = FCNImageLocalization().to(device)
model.load_state_dict(torch.load("differentscales.pth"))  # Adjust file path
# model.load_state_dict(torch.load("customcnnmodel.pth"))  # Adjust file path
model.eval()  # Set to evaluation mode

# Define transformation (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
])

# Load example maps
map1 = Image.open("mapfiles/output/0_shifted.png").convert("RGB")

# Apply transformation
map1 = transform(map1).unsqueeze(0).to(device)  # Add batch dimension

# Function to visualize feature maps
def visualize_feature_maps(model, map_input, layer):
    with torch.no_grad():
        x = map_input
        for name, module in model.named_children():
            x = module(x)  # Forward pass through each layer
            if name == layer:  # Stop at the specified layer
                break

    # Convert feature maps to numpy
    feature_maps = x.squeeze(0).cpu().numpy()

    # Plot feature maps
    num_filters = feature_maps.shape[0]
    fig, axes = plt.subplots(1, min(num_filters, 6), figsize=(15, 5))  # Show 6 feature maps
    for i in range(min(num_filters, 6)):
        # axes[i].imshow(feature_maps[i], cmap="viridis")
        # axes[i].axis("off")
        # axes[i].set_title(f"Filter {i+1}")


        axes.imshow(feature_maps[i], cmap="viridis")
        axes.axis("off")
        axes.set_title(f"Filter {i+1}")

    plt.show()

# Visualize feature maps from different layers
visualize_feature_maps(model, map1, "conv1")  # First conv layer
visualize_feature_maps(model, map1, "conv2")  # Second conv layer
visualize_feature_maps(model, map1, "conv3")  # Third conv layer
# visualize_feature_maps(model, map1, "conv4")  # Third conv layer