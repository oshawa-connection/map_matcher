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
        self.poly.AddGeometry(ring)

    def overlapsOtherExtent(self, other: "Extent") -> bool:
        return self.poly.Intersects(other.poly)

    def isDisjoint(self, other: "Extent") -> bool:
        return not self.overlapsOtherExtent(other)

    def isFullyWithin(self, other: "Extent"):
        return self.poly.Within(other.poly)

    def createRandomExtentWithinThisExtent(self, percentageToCover: float):

        def getMaximumFromPercentage(minD, maxD, percent):
            return (maxD - minD) * percent

        if not (0 <= percentageToCover <= 1): 
            raise ValueError('Percent must be between 0 and 1')
        pass

    def translatedExtent(self, dx:float, dy:float) -> "Extent":
        return Extent(self.extentMinX + dx, self.extentMaxX+dx, self.extentMinY+dy, self.extentMaxY+dy)
    
    def toArguments(self):
        return [str(self.extentMinX), str(self.extentMinY), str(self.extentMaxX), str(self.extentMaxY)]