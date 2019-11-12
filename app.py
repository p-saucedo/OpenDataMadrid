import geopy
import pandas
from geopy.geocoders import Nominatim, GoogleV3
from addressFixer import AddresFixer
import json
from geoJSONMaker import GeoJSONMaker

# Use geojson.io

def main():
    io = pandas.read_csv('datasets/A2.csv', index_col=None, header=0, sep=";")

    geolocator = Nominatim(user_agent="myGeocoder", timeout=5)
    #geolocator = GoogleV3()
    
    def get_latitude(x):
        if hasattr(x,'latitude') and (x.latitude is not None): 
            return x.latitude
        # Si no se ha encontrado la direccion, la latitud sera 0
        else:
            return 0
        

    def get_longitude(x):
        if hasattr(x,'longitude') and (x.longitude is not None):
            return x.longitude
        # Si no se ha encontrado la direccion, la longitud sera 0
        else:
            return 0
        

    # Rellenamos con las columnas Country y City para mejorar la geolocalizacion
    io['Country'] = 'Spain'
    io['City']  = 'Madrid'
    io['FixedAddress'] = io['CALLE'].apply(AddresFixer.fixAddress)
    io['FullAddress'] = io['FixedAddress'].map(str) + " " + io['City'].map(str) + " " + io['Country'].map(str)
    
    # Geolocalizamos las direcciones
    print("Geolocalizando las direcciones...")
    geolocate_column = io['FullAddress'].apply(geolocator.geocode)
    print(geolocate_column)
    
    # Rellenamos las columnas de longitud y latitud
    print("Seleccionando coordenadas...")
    io['latitude']= geolocate_column.apply(get_latitude)
    io['longitude'] = geolocate_column.apply(get_longitude)
   
    io.to_csv('geo_out.csv')
    geo = GeoJSONMaker()
    geo.CSVtoGeoJSON(csv_file='geo_out.csv')
   
if __name__ == "__main__":
    main()
