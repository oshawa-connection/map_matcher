import pandas as pd
import json

def bbox_to_polygon(minX, maxX, minY, maxY):
    return [
        [minX, minY],
        [maxX, minY],
        [maxX, maxY],
        [minX, maxY],
        [minX, minY]
    ]

df = pd.read_csv("/home/james/Documents/fireHoseSam/mapfiles/subgrid/tofindmetadata.csv")  # Change filename if needed

minX = df['minX'].min()
maxX = df['maxX'].max()
minY = df['minY'].min()
maxY = df['maxY'].max()

polygon = bbox_to_polygon(minX, maxX, minY, maxY)

feature = {
    "type": "Feature",
    "properties": {},
    "geometry": {
        "type": "Polygon",
        "coordinates": [polygon]
    }
}

geojson = {
    "type": "FeatureCollection",
    "features": [feature]
}

with open("bbox_extent.geojson", "w") as f:
    json.dump(geojson, f, indent=2)