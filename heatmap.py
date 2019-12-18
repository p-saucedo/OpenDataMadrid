from flask import Flask
import pandas as pd
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime

app = Flask(__name__)

@app.route('/')
def index():
    # Coordenadas del centro del mapa
    start_coords = (40.416775, -3.703790)
    # Creando el mapa
    folium_map = folium.Map(location=start_coords, 
                            zoom_start=14)

    # Leamos el dataset
    df = pd.read_csv("out_csv/geo_out2.csv", delimiter=';')
    df['count'] = 1
    # Create an object sorted by timestamp in case chronological evolution is desired.
    # df_hour_list = []
    # for hour in df.HORA.sort_values().unique():
    #     df_hour_list.append(df.loc[df.HORA == hour, ['latitude', 'longitude', 'count']]\
    #         .groupby(['latitude', 'longitude']).sum().reset_index().values.tolist())

    # Generamos el mapa de calor responsivo
    HeatMap(data=df[['latitude', 'longitude', 'LESIVIDAD*']].groupby(['latitude', 'longitude']).sum()\
                                                            .reset_index().values.tolist(), radius=8, max_zoom=13)\
                                                            .add_to(folium_map)
    
    # HeatMapWithTime(df_hour_list, radius=8, min_opacity=0.5, max_opacity=0.8, auto_play=True).add_to(folium_map)
    print(folium_map._repr_html_())
    return folium_map._repr_html_()


if __name__ == '__main__':
    app.run(debug=True, threaded=True)