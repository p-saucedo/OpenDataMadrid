window.onload = function () {
    var basemap = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
		attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
	});

    $.getJSON("map.geojson", function(data) {

    var geojson = L.geoJson(data, {
      onEachFeature: function (feature, layer) {
        layer.bindPopup(feature.properties.CALLE);
      }
    });


    var map = L.map('my-map')
    .fitBounds(geojson.getBounds());


    var popup = L.popup();
    function onMapClick(e) {
      popup
        .setLatLng(e.latlng)
        .setContent("Has clickado en " + e.latlng.toString())
        .openOn(map);
    }

    var DebugGrid = VirtualGrid.extend({
      options: {
        cellSize: 64,
        pathStyle: {
          color: '#3ac1f0',
          weight: 2,
          opacity: 0.5,
          fillOpacity: 0.25
        }
      },
      initialize: function  (options) {
        L.Util.setOptions(this, options);
        this.rects = {};
      },
      createCell: function (bounds, coords) {
        this.rects[this.coordsToKey(coords)] = L.rectangle(bounds, this.options.pathStyle).addTo(map);
      },
      cellEnter: function (bounds, coords) {
        var rect = this.rects[this.coordsToKey(coords)];
        map.addLayer(rect);
      },
      cellLeace: function (bounds, coords) {
        var rect = this.rects[this.coordsToKey(coords)];
        map.removeLayer(rect);
      },
      coordsToKey: function (coords) {
        return coords.x + ':' + coords.y + ':' +coords.z;
      }
    });
    var debugGrid = new DebugGrid().addTo(map);
    map.on('click', onMapClick);
    basemap.addTo(map);
    geojson.addTo(map);
  });
  
  
};



