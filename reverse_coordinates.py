import geopy
import pandas
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Bing


class Reverse:
    def __init__(self):
         # Used for Bing
        key = 'Ah274jwZ6fNiPYZQBTIoyfaV50oTOmCNlMz5RpJGulTHkvdifuvqA1xLCNmiFfDe'
        
        self.geolocator = Bing(api_key=key, timeout=7)


    def reverse_coord(self, latitude, longitude):
        try:
            addr = self.geolocator.reverse(query=(latitude, longitude))
        except GeocoderTimedOut:
            addr = 'Not working'

        addr = addr.address.split('Madrid')
        addr = addr[0].replace(',', '')
        return addr


