from typing import Tuple
import random
import attrs
from attrs import define
from osgeo import ogr

@define
class Extent:

    extentMinX: float
    extentMaxX: float
    extentMinY: float
    extentMaxY: float
    poly: ogr.Geometry = attrs.field(init=False)

    def __attrs_post_init__(self):
        # Create a ring from the extent coordinates
        ring = ogr.Geometry(ogr.wkbLinearRing)
        ring.AddPoint(self.extentMinX, self.extentMinY)
        ring.AddPoint(self.extentMaxX, self.extentMinY)
        ring.AddPoint(self.extentMaxX, self.extentMaxY)
        ring.AddPoint(self.extentMinX, self.extentMaxY)
        ring.AddPoint(self.extentMinX, self.extentMinY)

        # Create polygon
        self.poly = ogr.Geometry(ogr.wkbPolygon)
        # use to set spatial filter
        self.poly.AddGeometry(ring)

    def getDistanceX(self):
        return self.extentMaxX - self.extentMinX
    
    def getDistanceY(self):
        return self.extentMaxY - self.extentMinY

    def overlapsOtherExtent(self, other: "Extent") -> bool:
        return self.poly.Intersects(other.poly)

    def isDisjoint(self, other: "Extent") -> bool:
        '''
        Use to check if "no match" pair extents are valid
        '''
        return not self.overlapsOtherExtent(other)

    def isFullyWithin(self, other: "Extent"):
        '''
        Use to check if the smaller random/ shifted extent are within the whole area extent.
        '''
        return self.poly.Within(other.poly)

    def normaliseRelativeToOther(self, other:"Extent") -> Tuple[float,float,float,float]:
        def normalize(value, min_val, max_val):
            """Returns how far value is between min_val and max_val as a percentage (0 to 1)."""
            return (value - min_val) / (max_val - min_val)
        
        minX = normalize(self.extentMinX, other.extentMinX, other.extentMaxX)
        maxX = normalize(self.extentMaxX, other.extentMinX, other.extentMaxX)

        minY = normalize(self.extentMinY, other.extentMinY, other.extentMaxY)
        maxY = normalize(self.extentMaxY, other.extentMinY, other.extentMaxY)

        return minX, minY, maxX, maxY

    def createRandomExtentWithinThisExtent(self, percentageToCover: float):

        def getMaximumFromPercentage(minD, maxD, percent):
            return (maxD - minD) * percent

        if not (0 <= percentageToCover <= 1): 
            raise ValueError('Percent must be between 0 and 1')
        
        distXWithShift = getMaximumFromPercentage(self.extentMinX, self.extentMaxX, percentageToCover)
        distYWithShift = getMaximumFromPercentage(self.extentMinY, self.extentMaxY, percentageToCover)
        
        
        startX = random.uniform(self.extentMinX, self.extentMaxX - distXWithShift)
        startY = random.uniform(self.extentMinY, self.extentMaxY - distYWithShift)

        return Extent(startX, startX + distXWithShift,startY, startY + distYWithShift)

    def createRandomClippingExtent(self,percentOverlap = 0.2):
        '''
        keep this crude for now, always shift top right direction
        '''
        minClipX = self.getDistanceX() * percentOverlap + self.extentMinX
        minClipY = self.getDistanceY() * percentOverlap + self.extentMinY


        maxClipX = self.extentMaxX - self.getDistanceX() * percentOverlap
        maxClipY = self.extentMaxY - self.getDistanceY() * percentOverlap

        
        startX = random.uniform(minClipX, maxClipX)
        startY = random.uniform(minClipY, maxClipY)

        return Extent(startX, startX + self.getDistanceX(),startY, startY + self.getDistanceY())

    def createRandomDisjointExtent(self):
        '''
        keep this crude for now, always shift top right direction
        '''
        return Extent(self.extentMaxX, self.extentMaxY,self.extentMaxX + self.getDistanceX(), self.extentMaxY + self.getDistanceY())

    def translatedExtent(self, dx:float, dy:float) -> "Extent":
        return Extent(self.extentMinX + dx, self.extentMaxX+dx, self.extentMinY+dy, self.extentMaxY+dy)
    
    def toArguments(self):
        return [str(self.extentMinX), str(self.extentMinY), str(self.extentMaxX), str(self.extentMaxY)]
    


if __name__ == '__main__':
    from osgeo import gdal
    gdal.UseExceptions()

    wholeextent = Extent(0,100,0,100)

    wholeextent.createRandomExtentWithinThisExtent(10)


    print('hello')