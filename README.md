# Map Matcher

## Brief introduction

For training (note that this used to be done in Python in the `mapfiles` directory, but has now been moved [here](https://github.com/oshawa-connection/sharpFirehose))
1. Generate image pairs. A pair can either be matching or disjoint. 
2. For a matching pair
    - Find a random small area of the training basemap with at least X features large enough to be distinctive for your kernel size.
    - Shift the area by a small percentage in a random direction and random amount.
    - If the shifted area and the original area have enough large features in common, then draw both images. Otherwise try again with a different area.

3. For a disjoint pair, 
    - Find a random small area of the training basemap with at least X features large enough to be distinctive for your kernel size.
    - Find another random small area of the training basemap.
    - If the shifted area and the original area have some features in common, don't draw. Otherwise draw!
 
4. At the end of training dataset generation, you should have a set of image pairs, and a CSV like this:

```csv
image1Path, image2Path, matches
image1.png, image2.png, True
image3.png, image4.png, False
...
```

5. Run `python3 classifyPrecision.py`



Then for searching:
1. Given a PDF map of an area, split it up into 128 * 128 pixel tiles to allow them to fit on the GPU with a decent batch size.
2. Offset the grid, then draw the paper map tile grid again.
3. For the search area, making sure to keep roughly the same scale, draw 128 * 128 tiles. 
4. Offset the search area grid and do the same again.
Now run `multiGridSearch.py` which:
4. For each PDF tile, matches it against all of the search tiles. Does the same with the offset grids.
5. In theory, you should identify clusters of matched areas. Given that you know the arrangement of the PDF tiles relative to each other,
you should hopefully be able to spot the area that the PDF map represents of the search area.



## How to create synthetic dataset

1. Clone Mapserver into root directory.
2. Download OSM buildings dataset from geofabrik, process into a FGB and place under mapfiles/data

```sh
docker run -v ./mapfiles:/mapfiles --rm -it --entrypoint /bin/bash firehosesam
python3 /mapfiles/firehose.py
```

## How to train model

```sh
python3 classifyPrecision.py
```


The results of the parameter grid search favour this parameter combo:

```python
combo = {
    'nlayers': 4,
    'downSample': None,
    'leaky_cnn': True,
    'leaky_classifier': True,
    'base_channels': 32, 
    'padding': 0,
    'classifier_layers': 4,
    'classifier_hidden': 32,
    'learning_rate': 1e-3
}
```

And if that doesn't fit on your GPU, this is also a good parameter combo:
```python
combo = {
    'nlayers': 3,
    'downSample': None,
    'leaky_cnn': True,
    'leaky_classifier': False,
    'base_channels': 16, 
    'padding': 0,
    'classifier_layers': 4,
    'classifier_hidden': 128,
    'learning_rate': 1e-3
}
```

## Training + grid generation

- Training + testing images
- When generating the training/ test dataset, we make sure that the sharedPixelArea matches the model kernel size.
- Generate training data using Manchester, Leeds and Sheffield, then validate against Birmingham.
- Calculate a "paint coverage" factor rather than using the number of features.
- Test the current best model against its own validation dataset to see which ones it got wrong. Might show what the problem is!

## Grid matching

- When doing matching, after a few tiles have been matched up, when deciding where the next tile should start searching, it would be more efficient to check the tiles including and around where the initial clusters are.
- If two tiles are blank, skip matching them? Or just skip classifying, and match.
- Offset the main grid and search again using that - this will help at tile boundaries.

## Extracting data

We want to transform the vector data into a format that mapserver can read fast during rendering. You can use any method you like, but flatgeobuf was
found to have very good ready performance!

```bash
osmium tags-filter south-yorkshire-latest.osm.pbf a/building -o jjj.osm
ogr2ogr buildings.fgb jjj.osm
```