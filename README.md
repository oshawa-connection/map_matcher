# Scuba diving; it sucks!
docker run --rm -it --entrypoint /bin/bash firehosesam

## How to create synthetic dataset

1. Clone Mapserver into root directory.
2. Download OSM buildings dataset from geofabrik, process into a FGB and place under mapfiles/data

```sh
docker run -v ./mapfiles:/mapfiles --rm -it --entrypoint /bin/bash firehosesam
python3 /mapfiles/firehose.py
```

## How to train model

```sh
python3 aiWillDoomUsAll.py
```


# TODO list

## Learning rate scheduler reload checkpoint

## Training + grid generation

- Generate training data using Manchester, Leeds and Sheffield, then validate against Birmingham.
- Calculate a "paint coverage" factor rather than using the number of features.
- Test the current best model against its own validation dataset to see which ones it got wrong. Might show what the problem is!

## Grid matching

- When doing matching, after a few tiles have been matched up, when deciding where the next tile should start searching, it would be more efficient to check the tiles including and around where the initial clusters are.
- Generate a single blank tile and look it up rather than generate lots of blank tiles (taking up load of space + CPU time to generate nothing)
- If two tiles are blank, skip matching them? Or just skip classifying, and match.
- Offset the main grid and search again using that - this will help at tile boundaries.

## Extracting data

```bash
osmium tags-filter south-yorkshire-latest.osm.pbf a/building -o jjj.osm
ogr2ogr buildings.gpkg jjj.osm
```