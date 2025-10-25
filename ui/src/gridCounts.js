import 'ol/ol.css';
import { Map, View } from 'ol';
import GeoJSON from 'ol/format/GeoJSON';
import { Vector as VectorLayer } from 'ol/layer';
import { OSM, Vector as VectorSource } from 'ol/source';
import { Style, Fill, Stroke } from 'ol/style';
import { fromExtent } from 'ol/geom/Polygon';
import { bbox as bboxStrategy } from 'ol/loadingstrategy';
import { transformExtent } from 'ol/proj';
import TileLayer from 'ol/layer/Tile';

console.log('hello world')

// URL of the GeoJSON
const geojsonUrl = '/data/grid_counts.geojson';

// Create a global reference to vector source and layer
let vectorSource, vectorLayer, allFeatures = [];

// Set up the map
const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({source: new OSM()})
  ],
  view: new View({
    center: [0, 0],
    zoom: 2
  })
});

// Helper: style by count
function getFeatureStyle() {
  return new Style({
    fill: new Fill({
      color: 'rgba(0, 123, 255, 0.5)'
    }),
    stroke: new Stroke({
      color: '#007bff',
      width: 1
    })
  });
}

// Filter features by count and update the layer
function filterFeatures(minCount, maxCount) {
  const filteredFeatures = allFeatures.filter(f => {
    const count = f.get('count');
    return count >= minCount && count <= maxCount;
  });

  vectorSource.clear();
  vectorSource.addFeatures(filteredFeatures);
}

// Load GeoJSON and setup everything
fetch(geojsonUrl)
  .then(res => res.json())
  .then(data => {
    const format = new GeoJSON();
    allFeatures = format.readFeatures(data, {
      dataProjection: 'EPSG:3857',
      featureProjection: 'EPSG:3857'
    });

    // Determine min/max count
    const counts = allFeatures.map(f => f.get('count'));
    const minCount = Math.min(...counts);
    const maxCount = Math.max(...counts);

    // Inject slider limits
    const slider = document.getElementById('count-slider');
    slider.min = minCount;
    slider.max = maxCount;
    slider.value = maxCount;
    

    slider.addEventListener('input', () => {
      filterFeatures(parseInt(slider.value), maxCount);
      console.log(slider.value);
    });

    vectorSource = new VectorSource();
    vectorSource.addFeatures(allFeatures);

    vectorLayer = new VectorLayer({
      source: vectorSource,
      style: getFeatureStyle()
    });

    map.addLayer(vectorLayer);

    // Fit view to features
    const extent = vectorSource.getExtent();
    map.getView().fit(extent, { padding: [20, 20, 20, 20] });
  });