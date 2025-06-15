'''
THIS MUST RUN AS PART OF THE DOCKER CONTAINER, NOT LOCALLY 
'''
from math import floor
import csv
import random
from osgeo import ogr, gdal
from extent import Extent
from runMapserverSubprocess import runMapserverSubprocess

gdal.UseExceptions() 
imageSizePixels = 1000
numberOfImagesToCreate = 10_000
minimumNumberOfFeatures = 30
intersectionThreshold = 0.1 # percentage (max 1) of features in original image that must be in shifted image for the shift pair to be accepted.

the_path = "/mapfiles/data/leeds_buildings.fgb"
driver = ogr.GetDriverByName("FlatGeoBuf")
dataSource = driver.Open(the_path, 0)

src_lyr = dataSource.GetLayer()
totalFeatureCount = src_lyr.GetFeatureCount()

wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY = src_lyr.GetExtent()
wholeAreaExtent = Extent(wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY)


mem_driver = ogr.GetDriverByName("Memory")
mem_ds = mem_driver.CreateDataSource("in_memory")
layer = mem_ds.CreateLayer("mem_layer", geom_type=src_lyr.GetGeomType(), srs=src_lyr.GetSpatialRef())

# Copy fields
src_layer_defn = src_lyr.GetLayerDefn()
for i in range(src_layer_defn.GetFieldCount()):
    field_defn = src_layer_defn.GetFieldDefn(i)
    layer.CreateField(field_defn)

# Copy features
for feat in src_lyr:
    new_feat = ogr.Feature(layer.GetLayerDefn())
    new_feat.SetFrom(feat)
    layer.CreateFeature(new_feat)
    new_feat = None  # Clean up

# Reset reading to start using the memory layer
layer.ResetReading()



def extentHasEnoughFeatures(extent: Extent, layer, minimumNumberOfFeatures: int) -> bool:
    layer.SetSpatialFilter(extent.poly)
    startingFeatureCount = layer.GetFeatureCount()
    layer.SetSpatialFilter(None)
    layer.ResetReading()

    return startingFeatureCount > minimumNumberOfFeatures


def getFeatureSetForExtent(extent: Extent, layer)-> set:
    startingIdSet = set()
    layer.SetSpatialFilter(extent.poly)
    for feature in layer:
        id = feature.GetField("osm_id")
        startingIdSet.add(id)

    layer.SetSpatialFilter(None)
    layer.ResetReading()

    return startingIdSet

i = 0
with open('/mapfiles/output/metadata.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['tileImagePath', 'inputImagePath', 'doTheyMatch'])
    shouldMatch = True

    while i < numberOfImagesToCreate:
        percentOfImageToDisplay = random.uniform(0.1,0.3)
        startingExtent = wholeAreaExtent.createRandomExtentWithinThisExtent(percentOfImageToDisplay)
        
        if not extentHasEnoughFeatures(startingExtent, layer, minimumNumberOfFeatures):
            continue

        startingExtentIds = getFeatureSetForExtent(startingExtent, layer)

        
        if shouldMatch is True:
            isMatching = 1
            # then match
            subPercent = random.uniform(0.1, 0.2)
            otherExtent = startingExtent.createRandomClippingExtent(subPercent)
            if not extentHasEnoughFeatures(otherExtent, layer, minimumNumberOfFeatures):
                continue
            
            otherExtentIds = getFeatureSetForExtent(otherExtent, layer)

            if len(otherExtentIds) < (intersectionThreshold * len(startingExtentIds)):
                print('Not enough shared features in clipping extent, skipping.')
            
            shouldMatch = False
        else:
            isMatching = 0
            otherExtent = startingExtent.createRandomDisjointExtent()
            
            if not extentHasEnoughFeatures(otherExtent, layer, minimumNumberOfFeatures):
                print('Disjoint extent does not have enough features, skipping')
                continue
            
            disjointExtentIds = getFeatureSetForExtent(otherExtent, layer)
            
            if not startingExtentIds.isdisjoint(disjointExtentIds):
                print('Disjoint extent shared some features with starting extent, skipping.')
                continue
            
            shouldMatch = True

        
            
        tileImageName = f'/mapfiles/output/{i}_tile.png'
        inputFilename = f'/mapfiles/output/{i}_input_{'match' if isMatching == 1 else 'disjoint'}.png'

        runMapserverSubprocess(tileImageName, startingExtent, imageSizePixels)
        runMapserverSubprocess(inputFilename, otherExtent, imageSizePixels)
        
        spamwriter.writerow([tileImageName,inputFilename,isMatching])
        print(f"Processed image pair {i}")
        
        i = i+1