import geopy
import pandas
from geopy.geocoders import Nominatim, GoogleV3
# Use geojson.io
def main():
    io = pandas.read_csv('census_last.csv', index_col=None, header=0, sep=",")
    geolocator = Nominatim(user_agent="myGeocoder", timeout=5)
    #geolocator = GoogleV3()
    def get_latitude(x):
        return x.latitude

    def get_longitude(x):
        return x.longitude

    io['helper'] = io['Area_Name'].map(str) + " " + io['Country'].map(str)
    geolocate_column = io['helper'].apply(geolocator.geocode)
    io['latitude'] = geolocate_column.apply(get_latitude)
    io['longitude'] = geolocate_column.apply(get_longitude)
    io.to_csv('geocoding-output.csv')


if __name__ == "__main__":
    main()