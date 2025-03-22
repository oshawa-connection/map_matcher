from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import shapely
from shapely.geometry import box
import geopandas as gpd

class RasterVectorBBoxDataset(Dataset):
    def __init__(self, csv_path, vector_path, image_transform=None, max_coords=2048):
        self.df = pd.read_csv(csv_path)
        self.vector_data = gpd.read_file(vector_path)
        self.transform = image_transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.max_coords = max_coords

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['imagePath']).convert('RGB')
        img = self.transform(img)

        # Bounding box
        minx, miny, maxx, maxy = row['minX'], row['minY'], row['maxX'], row['maxY']
        bbox = box(minx, miny, maxx, maxy)

        # Clip vector data to bbox
        clipped = self.vector_data[self.vector_data.geometry.intersects(bbox)].copy()
        clipped['geometry'] = clipped.geometry.intersection(bbox)

        # Flatten geometries to coordinate sequences
        coords = []
        for geom in clipped.geometry:
            if geom.is_empty:
                continue
            if geom.geom_type == 'Polygon':
                coords.extend(list(geom.exterior.coords))
            elif geom.geom_type == 'LineString':
                coords.extend(list(geom.coords))
            # Normalize to bbox origin
        coords = [(x - minx, y - miny) for x, y in coords]

        # Pad or truncate to fixed length
        coords_tensor = torch.zeros((self.max_coords, 2))
        for i, (x, y) in enumerate(coords[:self.max_coords]):
            coords_tensor[i, 0] = x
            coords_tensor[i, 1] = y

        # Target bbox as tensor
        target = torch.tensor([minx, miny, maxx, maxy], dtype=torch.float32)

        return img, coords_tensor, target



class RasterVectorBBoxModel(nn.Module):
    def __init__(self, coord_dim=2, max_coords=2048, hidden_dim=256):
        super().__init__()
        # Raster encoder (ResNet backbone)
        resnet = models.resnet18(pretrained=True)
        self.raster_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC
        self.raster_proj = nn.Linear(512, hidden_dim)

        # Vector encoder (simple MLP over coords)
        self.vector_encoder = nn.Sequential(
            nn.Linear(coord_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        # Head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Output bbox
        )

    def forward(self, image, coords):
        # Encode raster
        x_raster = self.raster_encoder(image).squeeze(-1).squeeze(-1)
        x_raster = self.raster_proj(x_raster)

        # Encode vector (B x N x 2)
        x_vector = self.vector_encoder(coords)  # B x N x H
        x_vector = x_vector.permute(0, 2, 1)  # B x H x N
        x_vector = self.pool(x_vector).squeeze(-1)  # B x H

        # Combine
        x = torch.cat([x_raster, x_vector], dim=1)
        return self.head(x)




dataset = RasterVectorBBoxDataset("bbox_labels.csv", "vector_data.geojson")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = RasterVectorBBoxModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()  # Or HuberLoss

for epoch in range(10):
    for img, coords, target in dataloader:
        pred = model(img, coords)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")