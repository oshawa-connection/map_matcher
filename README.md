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
## Grid matching

- When doing matching, after a few tiles have been matched up, when deciding where the next tile should start searching, it would be more efficient to check the tiles including and around where the initial clusters are.
- Generate a single blank tile and look it up rather than generate lots of blank tiles (taking up load of space + CPU time to generate nothing)
- If two tiles are blank, skip matching them? Or just skip classifying, and match.