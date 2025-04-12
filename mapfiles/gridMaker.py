from math import ceil
from osgeo import ogr, gdal
from extent import Extent
from runMapserverSubprocess import runMapserverSubprocess

imageDimensionPixels = 1000
percentOfImageToDisplay = 0.2

# the_path = "/mapfiles/data/leeds_buildings.fgb"
the_path= "/home/james/Documents/fireHoseSam/mapfiles/data/leeds_buildings.fgb"
driver = ogr.GetDriverByName("FlatGeoBuf")
dataSource = driver.Open(the_path, 0)

layer = dataSource.GetLayer()
totalFeatureCount = layer.GetFeatureCount()

wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY = layer.GetExtent()
wholeAreaExtent = Extent(wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY)

gridXStep = wholeAreaExtent.getDistanceX() / percentOfImageToDisplay
gridYStep = wholeAreaExtent.getDistanceY() / percentOfImageToDisplay

currentMax = percentOfImageToDisplay
currentExtent = Extent()

while currentExtent.

for i in range(0,1, 0.2):
    print(i)


# while currentMax < 1 + percentOfImageToDisplay:
#     print(currentMax)
#     currentMax = currentMax + percentOfImageToDisplay
    

# Extent(wholeAreaExtent.extentMinX)

