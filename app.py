import geopy
import pandas
from geopy.geocoders import Nominatim, GoogleV3
from parser_dir import ParserDir

# Use geojson.io

def main():
    io = pandas.read_csv('datasets/A2.csv', index_col=None, header=0, sep=";")
    geolocator = Nominatim(user_agent="myGeocoder", timeout=5)
    #geolocator = GoogleV3()
    def get_latitude(x):
        return x.latitude

    def get_longitude(x):
        return x.longitude

    io['Country'] = 'Spain'
    io['City']  = 'Madrid'
    io['CALLE_BIEN'] = io['CALLE'].apply(ParserDir.parsearDireccion)
    io['helper'] = io['CALLE_BIEN'].map(str) + " " + io['City'].map(str) + " " + io['Country'].map(str)
    geolocate_column = io['helper'].apply(geolocator.geocode)
    #geolocate_column = io['DireccionCompleta'].apply(geolocator.geocode)
    print(geolocate_column)
    
    #io['latitude']= geolocate_column.apply(get_latitude)
    #io['longitude'] = geolocate_column.apply(get_longitude)
   
    io.to_csv('geocoding-output.csv')

if __name__ == "__main__":
    main()
