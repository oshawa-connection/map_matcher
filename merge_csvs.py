import pandas as pd

# Load CSVs
csv1 = pd.read_csv("/home/james/Documents/fireHoseSam/image_pair_results.csv")
grid = pd.read_csv("/home/james/Documents/fireHoseSam/mapfiles/gridsearch/gridmetdata.csv")
sub_grid = pd.read_csv("/home/james/Documents/fireHoseSam/mapfiles/subgrid/tofindmetadata.csv")

# Merge grid metadata
merged = csv1.merge(grid, left_on="grid_image", right_on="filepath", suffixes=('', '_grid'))
merged = merged.merge(sub_grid, left_on="sub_image", right_on="filepath", suffixes=('', '_sub'))

# Select desired columns
result = merged[[
    "grid_image", "sub_image", "probability",
    "minX", "maxX", "minY", "maxY", "isBlank",
    "minX_sub", "maxX_sub", "minY_sub", "maxY_sub", "isBlank_sub"
]]

# Write to CSV
result.to_csv("joined_results.csv", index=False)