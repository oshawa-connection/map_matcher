'''
THIS MUST RUN AS PART OF THE DOCKER CONTAINER, NOT LOCALLY 
'''
from math import floor
import multiprocessing
import concurrent.futures
import csv
import subprocess
import random

from osgeo import ogr, gdal
from extent import Extent
from mapfiles.runMapserverSubprocess import runMapserverSubprocess

gdal.UseExceptions() 
imageSizePixels = 1000
numberOfImagesToCreate = 10
minimumNumberOfFeatures = 30

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
    spamwriter.writerow(['starting', 'shifted', 'minX','minY', 'maxX', 'maxY'])
    while i < numberOfImagesToCreate:
        # choose a percentage of the image to display
        
        percentOfImageToDisplay = random.uniform(0.1,0.3)
        subImagePercentOfParent = random.uniform(0.3,0.7)
        
        startingExtent = wholeAreaExtent.createRandomExtentWithinThisExtent(percentOfImageToDisplay)
        if not extentHasEnoughFeatures(startingExtent, layer, 100):
            print('Starting extent did not contain enough features, skipping')
            continue

        
        shiftedExtent = startingExtent.createRandomExtentWithinThisExtent(subImagePercentOfParent)
        

        if not extentHasEnoughFeatures(shiftedExtent, layer, 30):
            print('Shifted extent did not contain enough features, skipping')
            continue

        startingFileName = f'/mapfiles/output/{i}_starting.png'
        runMapserverSubprocess(startingFileName, startingExtent, imageSizePixels)
        shiftedFileName = f'/mapfiles/output/{i}_shifted.png'
        runMapserverSubprocess(shiftedFileName, shiftedExtent, floor(imageSizePixels * subImagePercentOfParent))

        
        spamwriter.writerow([f'{i}_starting.png', f'{i}_shifted.png', *shiftedExtent.normaliseRelativeToOther(startingExtent)])

        i+=1

        

print('Completed successfully')