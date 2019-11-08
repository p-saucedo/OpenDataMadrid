import geopy 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import csv
from parser_dir import ParserDir
import json
from NominatimAPI import NominatimAPI

def main():
    i = 0
    # Leemos el csv
    with open('datasets/AccidentesBicicletas_2019.csv','r',encoding='utf-8', errors='ignore') as csvFile:
        csv_reader = csv.reader(csvFile, delimiter=';')
        line_count = 0
        direcciones = [] # Lista con todas las direcciones
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                dir = row[3]
                #print(f'\t{dir} es la calle del accidente.')
                direcciones.append(dir)
                line_count += 1
        print(f'Processed {line_count} lines.')

    p = ParserDir()
    for i in range(line_count-1):
        direcciones[i] = p.parsearDireccion(str(direcciones[i]))

    # USO DE GEOPY
    # En user_agent tiene que ir el nombre de la App
    ##locator = geopy.geocoders.Nominatim(user_agent="myGeocoder")
    # Uso para sacar coordenadas
    ##location = locator.geocode("CALLE DONOSO CORTES / CALLE ANDRES MELLADO,MADRID,SPAIN")
    # Uso para sacar ubicacion
    ###ubicacion = locator.reverse("40.4278811,-3.6892325")
    ##print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))

    # USO DE API de NOMINATIM
    api = NominatimAPI()

    lat,lon = api.getCoordenates("12 mesena","madrid","spain")
    print(lat)
    print(lon)



    #street_map = gpd.read_file('D05/02_barrio_l.shp')
    #fig, ax = plt.subplots(figsize = (15,15))
    #street_map.plot(ax = ax)


if __name__ == "__main__":
    main()