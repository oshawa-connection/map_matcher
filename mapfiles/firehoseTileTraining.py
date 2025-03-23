'''
THIS MUST RUN AS PART OF THE DOCKER CONTAINER, NOT LOCALLY 
'''
from math import floor
import asyncio
import multiprocessing
import concurrent.futures
import csv
import subprocess
import random

from osgeo import ogr, gdal
from extent import Extent
from runMapserverSubprocess import runMapserverSubprocess

gdal.UseExceptions() 
imageSizePixels = 1000
numberOfImagesToCreate = 5000
minimumNumberOfFeatures = 30
intersectionThreshold = 0.1 # percentage (max 1) of features in original image that must be in shifted image for the shift pair to be accepted.

the_path = "/mapfiles/data/leeds_buildings.fgb"
driver = ogr.GetDriverByName("FlatGeoBuf")
dataSource = driver.Open(the_path, 0)

layer = dataSource.GetLayer()
totalFeatureCount = layer.GetFeatureCount()

wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY = layer.GetExtent()
wholeAreaExtent = Extent(wholeAreaMinX, wholeAreaMaxX, wholeAreaMinY, wholeAreaMaxY)


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
            otherExtent = startingExtent.createRandomClippingExtent(0.2)
            if not extentHasEnoughFeatures(otherExtent, layer, minimumNumberOfFeatures):
                continue
            
            otherExtentIds = getFeatureSetForExtent(otherExtent, layer)

            if len(otherExtentIds) < (intersectionThreshold * len(startingExtentIds)):
                print('Not enough shared features in clipping extent, skipping.')

            print('generating a matching pair')
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
            
            print('generating a non-matching pair')
            shouldMatch = True

            
        tileImageName = f'/mapfiles/output/initial_tile_{i}.png'
        inputFilename = f'/mapfiles/output/input_image_{i}.png'

        runMapserverSubprocess(tileImageName, startingExtent, imageSizePixels)
        runMapserverSubprocess(inputFilename, otherExtent, imageSizePixels)
        
        spamwriter.writerow([tileImageName,inputFilename,isMatching])
        print(f"Processed image pair {i}")
        
        i = i+1