import pandas as pd
import json

def bbox_to_polygon(minX, maxX, minY, maxY):
    # Returns coordinates for a square polygon
    return [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
        [minX, minY]
    ]

df = pd.read_csv("joined_results.csv")  # Change to your CSV filename

features = []

for _, row in df.iterrows():
    # Grid polygon
    grid_coords = bbox_to_polygon(row['minX'], row['maxX'], row['minY'], row['maxY'])
    grid_feature = {
        "type": "Feature",
        "properties": {
            "type": "grid",
            "grid_image": row['grid_image'],
            "probability": row['probability'],
            "isBlank": row['isBlank']
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [grid_coords]
        }
    }
    features.append(grid_feature)

    # # Subgrid polygon
    # sub_coords = bbox_to_polygon(row['minX_sub'], row['maxX_sub'], row['minY_sub'], row['maxY_sub'])
    # sub_feature = {
    #     "type": "Feature",
    #     "properties": {
    #         "type": "subgrid",
    #         "sub_image": row['sub_image'],
    #         "probability": row['probability'],
    #         "isBlank_sub": row['isBlank_sub']
    #     },
    #     "geometry": {
    #         "type": "Polygon",
    #         "coordinates": [sub_coords]
    #     }
    # }
    # features.append(sub_feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

with open("output.geojson", "w") as f:
    json.dump(geojson, f, indent=2)