window.onload = function () {
    var basemap = L.tileLayer('http://{s}.tile.osm.org/{z}/{x}/{y}.png', {
		attribution: '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors'
	});
    var base = '127.0.0.1:5000';

    // Opciones para el disenio de los iconos
    var geojsonMarkerOptions = {
      radius: 4,
      fillColor: "#ff7800",
      color: "#000",
      weight: 1,
      opacity: 0,
      fillOpacity: 0
  };

    $.getJSON("/static/maps/map.geojson", function(data) {
    coords = []
    var geojson = L.geoJson(data, {
      // Para el disenio del icono
      pointToLayer: function (feature, latlng) {
          return L.circleMarker(latlng, geojsonMarkerOptions);
      },
      onEachFeature: function (feature, layer) {
        /*c = []
        c.push(feature.geometry.coordinates[1], feature.geometry.coordinates[0])
        coords.push(c)*/
        //layer.bindPopup(feature.properties.CALLE);
      }
    });


    // Para el mapa de calor del mapa en JS
    var heat = L.heatLayer(coords, {
      radius: 30,
      blur : 15,
      maxZoom: 12,
      max: 4.0,
      gradient: {
        0.4: 'yellow',
        0.7: 'orange',
        1.0: 'red'
    }
    });

    // Para el mapa en html
    $.ajax({
      url: '/get_heatmap',
      type: 'POST',
      success: function(response){
        $("#heatmap").html(response);
      },
      error: function(error){
        console.log(error);
      }
    });

    function changeOp(value){
      coords = []
      valueSum = parseInt(value) * 100;

      geojson.eachLayer(function(layer){
       
        horaTime = layer.feature.properties.HORA.split(':')
        horaSum = parseInt(horaTime[0]) * 100 + parseInt(horaTime[1])

        //if( horaSum >= valueSum && horaSum <(valueSum + 100)){
        if(horaSum <(valueSum + 100)){
          layer.setStyle({opacity : 1, fillOpacity: 0.8})
          c = []
          c.push(layer.feature.geometry.coordinates[1], layer.feature.geometry.coordinates[0])
          coords.push(c)
        }else{ // No la cumplen
          layer.setStyle({opacity : 0, fillOpacity: 0}) 
        }
      });

      layers = []
      geojson.eachLayer(function(layer){
        layers.push(layer);
        geojson.removeLayer(layer);
      });
      var i;
      for(i=0; i<layers.length; i++){
        geojson.addLayer(layers[i]);
      }
      heat.setLatLngs(coords);
      heat.redraw();
    }

    console.log(geojson);

    var slider = L.control.slider(function(value) {
      changeOp(value);
    }, {
    min: 0,
    max: 23,
    value: 0,
    step:1,
    size: '250px',
    orientation:'horizontal',
    id: 'slider',
    logo: 'H',
    syncSlider: 'true'
    });

    var map = L.map('my-map')
    .fitBounds(geojson.getBounds());

    var popup = L.popup();
    function onMapClick(e) {
      /*popup
        .setLatLng(e.latlng)
        .setContent("Has clickado en " + e.latlng.toString())
        .openOn(map);*/
        var pos_dict = '{ "latitude": ' + e.latlng.lat + ',' +  '"longitude": ' + e.latlng.lng + '}';
        $.ajax({
          url: '/get_click',
			    data: JSON.parse(pos_dict),
			    type: 'POST',
			    success: function(response){
  
            var p = "Has clickado en " + e.latlng.lat + ", " + e.latlng.lng ;
            var l = "En caso de accidente, la gravedad seria de : " + response['lesividad'];
            var t = "Probabilidad de accidente: " + response['prob'] + " %";
    
            $("#subtitulo").html(p);
            $("#lesividad").html(l);
            $("#probabilidad").html(t);
		    	},
		  	  error: function(error){
				    console.log(error);
        }
      
        });        
    }

    
    map.on('click', onMapClick);
    heat.addTo(map);
    basemap.addTo(map);
    geojson.addTo(map);
    slider.addTo(map);
  });

  
};



