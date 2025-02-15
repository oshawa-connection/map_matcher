import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Define the model class (must match the saved model)
class MapBoundingBoxRegressor(nn.Module):
    def __init__(self):
        super(MapBoundingBoxRegressor, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=False)  # No need to load pretrained weights
        self.feature_extractor.fc = nn.Identity()
        
        self.regressor = nn.Sequential(
            nn.Linear(512 * 1, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, map1, map2):
        f1 = self.feature_extractor(map1)
        f2 = self.feature_extractor(map2)
        fused_features = torch.abs(f1 - f2)
        magicNumber = self.regressor(fused_features)
        return magicNumber

# Load the trained model
model = MapBoundingBoxRegressor()
model.load_state_dict(torch.load("model.pth"))
model.eval()

if not torch.cuda.is_available():
    raise Exception("No GPU")

# Move model to GPU if available
device = torch.device("cuda")
model.to(device)

# Define image transformations (must match training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

for i in range(0,1000):


    # Load two map images
    map1_path = f"mapfiles/output/{i}_shifted.png"  # Replace with actual path
    map2_path = f"mapfiles/output/{i}_starting.png"  # Replace with actual path

    map1 = Image.open(map1_path).convert("RGB")
    map2 = Image.open(map2_path).convert("RGB")

    # Apply transforms
    map1 = transform(map1).unsqueeze(0).to(device)  # Add batch dimension
    map2 = transform(map2).unsqueeze(0).to(device)

    # Make a prediction
    with torch.no_grad():
        magic_number_prediction = model(map1, map2)

    print(f"Predicted Magic Number {i}: {magic_number_prediction.item():.6f}")
