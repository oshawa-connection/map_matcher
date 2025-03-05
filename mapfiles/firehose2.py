'''
THIS MUST RUN AS PART OF THE DOCKER CONTAINER, NOT LOCALLY 
'''
import multiprocessing
import concurrent.futures
import csv
import subprocess
import random

from osgeo import ogr, gdal
from mapfiles.extent import Extent

gdal.UseExceptions() 

numberOfImagesToCreate = 10000
imageSizePixels = 1000.0 # keep things really simple for now, assume 1 pixel = 1 metre and perfectly square image.
minimumNumberOfFeatures = 30
intersectionThreshold = 0.1 # percentage (max 1) of features in original image that must be in shifted image for the shift pair to be accepted.

the_path = "/mapfiles/data/leeds_buildings.fgb"
driver = ogr.GetDriverByName("FlatGeoBuf")
dataSource = driver.Open(the_path, 0)

layer = dataSource.GetLayer()
totalFeatureCount = layer.GetFeatureCount()

wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY = layer.GetExtent()
wholeAreaExtent = Extent(wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY)


def getMaximumFromPercentage(minD, maxD, percent):
    return (maxD - minD) * percent

def boundsToWktPolygon(startX, startY, endX, endY):
    wkt= f"POLYGON (({startX} {startY}, {endX} {startY}, {endX} {endY},{startX} {endY},{startX} {startY}))"
    return ogr.CreateGeometryFromWkt(wkt)


def extentHasEnoughFeatures(extent: Extent, layer) -> bool:
    layer.SetSpatialFilter(extent.poly)
    startingFeatureCount = layer.GetFeatureCount()
    layer.SetSpatialFilter(None)
    layer.ResetReading()

    return startingFeatureCount > minimumNumberOfFeatures


def featureSetInExtent(extent: Extent, layer)-> set:
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
    spamwriter.writerow(['starting', 'shifted', 'xPixelShift','yPixelShift', 'match'])
    while i < numberOfImagesToCreate:
        # choose a percentage of the image to display
        
        percentOfImageToDisplay = random.uniform(0.01,0.2)
        percentShift = random.uniform(0.01,0.5)
        
        startingExtent = wholeAreaExtent.createRandomExtentWithinThisExtent(percentOfImageToDisplay)
        x_shift = startingExtent.getDistanceX() * percentShift
        y_shift = startingExtent.getDistanceY() * percentShift

        shiftedExtent = startingExtent.translatedExtent(x_shift, y_shift)

        xPixelsPerMetre = imageSizePixels / startingExtent.getDistanceX()
        yPixelsPerMetre = imageSizePixels / startingExtent.getDistanceY()

        if not extentHasEnoughFeatures(startingExtent, layer):
            print('starting extent did not contain enough features, skipping')
            continue

        startingSet = featureSetInExtent(startingExtent, layer)
        shiftedSet = featureSetInExtent(shiftedExtent, layer)

        minimumNumberOfSharedFeatures = len(startingSet) * intersectionThreshold

        if len(startingSet.intersection(shiftedSet)) < minimumNumberOfSharedFeatures:
            print('Found a shift that had too few features in common, skipping')
            continue

        i+=1

        

print('Completed successfully')