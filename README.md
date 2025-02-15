## Scuba diving; it sucks!
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