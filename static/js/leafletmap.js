window.onload = function () {
    var basemap = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
		attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
	});
    var base = '127.0.0.1:5000';
    $.getJSON("/static/maps/map.geojson", function(data) {
  
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
        var pos_dict = '{ "latitude": ' + e.latlng.lat + ',' +  '"longitude": ' + e.latlng.lng + '}';
        console.log(pos_dict)
        $.ajax({
          url: '/get_click',
			    data: JSON.parse(pos_dict),
			    type: 'POST',
			    success: function(response){
            console.log(response)
            var p = "Has clickado en " + e.latlng.lat + ", " + e.latlng.lng ;
            var l = "En caso de accidente, la gravedad seria de : " + response['lesividad'];
            console.log(l)
            $("#subtitulo").html(p);
            $("#lesividad").html(l);
		    	},
		  	  error: function(error){
				    console.log(error);
        }
      
        });        
    }

    
    map.on('click', onMapClick);
    basemap.addTo(map);
    geojson.addTo(map);
  });

  
};



