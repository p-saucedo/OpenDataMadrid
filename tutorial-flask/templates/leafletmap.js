window.onload = function () {
    var basemap = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
		attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
	});

    $.getJSON({{mapgeojson}}, function(data) {

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

    map.on('click', onMapClick);
    basemap.addTo(map);
    geojson.addTo(map);
  });
  
  
};



