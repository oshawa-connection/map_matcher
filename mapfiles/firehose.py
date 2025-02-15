'''

'''
import csv
import subprocess
import random
from osgeo import ogr, gdal

gdal.UseExceptions() 

numberOfImagesToCreate = 1000
imageSizePixels = 1000.0 # keep things really simple for now, assume 1 pixel = 1 metre and perfectly square image.
minimumNumberOfFeatures = 30
intersectionThreshold = 0.1 # percentage (max 1) of features in original image that must be in shifted image for the shift pair to be accepted.

the_path = "/mapfiles/data/leeds_buildings.fgb"
driver = ogr.GetDriverByName("FlatGeoBuf")
dataSource = driver.Open(the_path, 0)

layer = dataSource.GetLayer()
totalFeatureCount = layer.GetFeatureCount()

extentMinX, extentMaxX, extentMinY, extentMaxY = layer.GetExtent()


def getMaximumFromPercentage(minD, maxD, percent):
    return (maxD - minD) * percent

def boundsToWktPolygon(startX, startY, endX, endY):
    wkt= f"POLYGON (({startX} {startY}, {endX} {startY}, {endX} {endY},{startX} {endY},{startX} {startY}))"
    return ogr.CreateGeometryFromWkt(wkt)

i = 0
with open('/mapfiles/output/metadata.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(['starting', 'shifted', 'xPixelShift','yPixelShift'])
    while i < numberOfImagesToCreate:

        # choose a percentage of the image to display
        
        percentOfImageToDisplay = random.uniform(0.01,0.2)
        percentShift = random.uniform(0.01,0.5)
        
        distX = getMaximumFromPercentage(extentMinX, extentMaxX, percentOfImageToDisplay)
        xShift = distX * percentShift
        distXWithShift = getMaximumFromPercentage(extentMinX, extentMaxX, percentOfImageToDisplay + percentShift)

        distY = getMaximumFromPercentage(extentMinY, extentMaxY, percentOfImageToDisplay)
        yShift = distY * percentShift
        distYWithShift = getMaximumFromPercentage(extentMinY, extentMaxY, percentOfImageToDisplay + percentShift)

        xPixelsPerMetre = imageSizePixels / distX
        yPixelsPerMetre = imageSizePixels / distY

        xPixelShift = xPixelsPerMetre * xShift / imageSizePixels
        yPixelShift = yPixelsPerMetre * yShift / imageSizePixels

        startX = random.uniform(extentMinX, extentMaxX - distXWithShift)
        startY = random.uniform(extentMinY, extentMaxY - distYWithShift)

        endX = startX + distX
        endY = startY + distY

        shiftedStartX = startX + xShift
        shiftedStartY = startY + yShift

        shiftedEndX = endX + xShift
        shiftedEndY = endY + yShift

        assert startX < endX
        assert startY < endY

        assert startX > extentMinX
        assert startY > extentMinY 
        
        assert shiftedEndX < extentMaxX
        assert shiftedEndY < extentMaxY

        startingExtentPolygon = boundsToWktPolygon(startX, startY, endX, endY)
        
        layer.SetSpatialFilter(startingExtentPolygon)
        startingFeatureCount = layer.GetFeatureCount()

        if (startingFeatureCount < minimumNumberOfFeatures):
            print('starting extent did not contain enough features, skipping')
            continue

        startingIdSet = set()
        for feature in layer:
            id = feature.GetField("osm_id")
            startingIdSet.add(id)
            
        
        startingExtentPolygon = None
        layer.ResetReading()


        shiftedExtentPolygon = boundsToWktPolygon(shiftedStartX, shiftedStartY, shiftedEndX, shiftedEndY)
        layer.SetSpatialFilter(shiftedExtentPolygon)
        startingFeatureCount = layer.GetFeatureCount()

        if (startingFeatureCount < minimumNumberOfFeatures):
            print('shifted extent did not contain enough features, skipping')
            continue


        shiftedIdSet = set()
        for feature in layer:
            id = feature.GetField("osm_id")
            shiftedIdSet.add(id)


        minimumNumberOfSharedFeatures = startingFeatureCount * intersectionThreshold

        if len(startingIdSet.intersection(shiftedIdSet)) < minimumNumberOfSharedFeatures:
            print('Found a shift that had too few features in common, skipping')
            continue

        shiftedExtentPolygon = None
        layer.SetSpatialFilter(None)


        startingFileName = f'/mapfiles/output/{i}_starting.png'

        starting_run_result = subprocess.run([ 
            '/hello/MapServer/build/map2img', 
            '-s', 
            str(imageSizePixels), 
            str(imageSizePixels), 
            '-all_debug', 
            '5', 
            '-map_debug', 
            '5', 
            '-l', 
            'buildings', 
            '-m', 
            '/mapfiles/some.map', 
            '-conf', 
            '/mapfiles/config.map', 
            '-e',
            str(startX),
            str(startY),
            str(endX),
            str(endY),
            '-o',
            startingFileName ],stdout = subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        starting_run_result.check_returncode()
        
        shiftedFileName = f'/mapfiles/output/{i}_shifted.png'

        shifted_run_result = subprocess.run([ 
            '/hello/MapServer/build/map2img', 
            '-s', 
            str(imageSizePixels), 
            str(imageSizePixels), 
            '-all_debug', 
            '5', 
            '-map_debug', 
            '5', 
            '-l', 
            'buildings', 
            '-m', 
            '/mapfiles/some.map', 
            '-conf', 
            '/mapfiles/config.map', 
            '-e',
            str(shiftedStartX),
            str(shiftedStartY),
            str(shiftedEndX),
            str(shiftedEndY),
            '-o',
            shiftedFileName ],stdout = subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        shifted_run_result.check_returncode()
        spamwriter.writerow([f'{i}_starting.png', f'{i}_shifted.png', xPixelShift, yPixelShift])
        i+=1

print('Completed successfully')