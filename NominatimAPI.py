import requests
"""
API of Nominatim

Returns:
    Objetic NominatimAPi
    
"""
class NominatimAPI:
    base_url = "https://nominatim.openstreetmap.org"
    
    def __init__(self):
        pass
    """
    Function that uses Nominatim API to get the cordinates of a specific adress.
    
    Returns:
        At pos 0, the function returns latitude
        At pos 1, the function returns longitude
    """
    def getCoordenates(self, direccion, ciudad, pais):
        params = {
            "street" : direccion,
            "city" : ciudad,
            "country" : pais,
            "format" : "json"
        }

        url = self.base_url + "/search"

        response = requests.get(url,params = params)
        if(response.status_code == 200):
            return response.json()[0]['lat'], response.json()[0]['lon']
        else:
            return -1,-1
