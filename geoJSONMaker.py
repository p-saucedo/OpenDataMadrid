import csv
import json

class GeoJSONMaker:


    def __init__(self):
        pass

    def CSVtoGeoJSON(self, csv_file):
        row_count = 0
        with open(csv_file) as csv_f:
            csv_reader = csv.reader(csv_f, delimiter=';')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    columns = row
                    line_count+=1
                else:
                    row_count += 1
    
        split_col = columns
        
        # La cabecera del fichero GeoJSON
        output ='''{"type":"FeatureCollection","features":['''
        prop = '''{"type":"Feature","properties":{'''
        field = '''"%s":"%s",'''
        # Sin la coma final para la ultima propiedad
        field2 = '''"%s":"%s"'''
        point = '''"geometry":{"type":"Point","coordinates":[%f,%f]}'''
        end = ''']}'''
        prop_aux = prop
        with open(csv_file) as csv_f:
            line_count = 0
            csv_reader = csv.reader(csv_f, delimiter=';')
            for row in csv_reader:
                if line_count > 0:
                    split_row = row
                    prop_aux = prop
                    for i in range(len(split_col)-1): # Longitud y latitud se cargan a la vez
                        if i < (len(split_col)-2): # Los dos ultimos son latitud y longitud
                            if i != (len(split_col) - 3)  : # La ultima propiedad va sin coma
                                prop_aux = prop_aux + field % (split_col[i],split_row[i])
                            else:
                                prop_aux = prop_aux + field2 % (split_col[i],split_row[i])
                        else:
                            prop_aux = prop_aux + "},"
                            prop_aux = prop_aux + point % (float(split_row[i+1]), float(split_row[i]))
                            if line_count != row_count:
                                prop_aux = prop_aux + "},"
                            else:
                                prop_aux = prop_aux + "}"
                    line_count += 1
                    output += prop_aux
                else:
                    line_count += 1
            output += end

        with open("maps/map.geojson", "w") as f:
            f.write(output)
       