import geopy
import pandas
import json
import sys
import os
import csv
import xlrd
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim, GoogleV3, Bing
from addressFixer import AddresFixer
from geoJSONMaker import GeoJSONMaker
# Use geojson.io

class mainProc:

    def __init__(self,file, delimitador, columa_direccion):
        filename, file_extension = os.path.splitext(file)
        if file_extension == '.xlsx':
            dataset = self.csv_from_excel(filename=filename)
        else:
            dataset = file
        route = 'uploads/' + dataset
        io = pandas.read_csv(route, index_col=None, header=0, sep=delimitador)

        # Used for Bing
        key = 'Ah274jwZ6fNiPYZQBTIoyfaV50oTOmCNlMz5RpJGulTHkvdifuvqA1xLCNmiFfDe'
        
        #geolocator = Nominatim(user_agent="myGeocoder", timeout=5)
        #geolocator = GoogleV3()
        geolocator = Bing(api_key=key, timeout=7)
        
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
        print("Arreglando direcciones...")
        io['FixedAddress'] = io[columa_direccion].apply(AddresFixer.fixAddress)
        io['FullAddress'] = io['FixedAddress'].map(str) + " " + io['City'].map(str) + " " + io['Country'].map(str)
        try:
            # Geolocalizamos las direcciones
            print("Geolocalizando las direcciones...")
            geolocate_column = self.do_geocode(io['FullAddress'],geolocator)
        except GeocoderServiceError:
            print("Error con el geocoder")
            return

        print(geolocate_column)
        
        # Rellenamos las columnas de longitud y latitud
        print("Seleccionando coordenadas...")
        io['latitude']= geolocate_column.apply(get_latitude)
        io['longitude'] = geolocate_column.apply(get_longitude)
    
        print("Creando geoJSON...")
        io.to_csv('out_csv/geo_out.csv', sep=';')
        geo = GeoJSONMaker()
        geo.CSVtoGeoJSON(csv_file='out_csv/geo_out.csv')

    # This function is used to avoid GeoCoderTimeOut, when we get that exception it tries again
    def do_geocode(self,addresses, geolocator):
        try:
            return addresses.apply(geolocator.geocode)
        except GeocoderTimedOut:
            return self.do_geocode(addresses,geolocator)

    def csv_from_excel(self,filename):

        f_xlsx = 'datasets/' + filename + '.xlsx'
        wb = xlrd.open_workbook(f_xlsx)
        sheet_names = wb.sheet_names()
        sh = wb.sheet_by_name(sheet_names[0])

        f_csv = 'datasets/' + filename + '.csv'
        your_csv_file = open(f_csv, 'w')
        wr = csv.writer(your_csv_file, delimiter=';')

        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))

        your_csv_file.close()
        ret = filename + '.csv'
        return ret
