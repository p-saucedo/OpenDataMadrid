from flask import Flask, render_template, request, redirect, jsonify
from werkzeug import secure_filename
from mainProc import mainProc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from engine import Engine as eng
import os
import numpy as np
import folium
from folium.plugins import HeatMap
from folium.plugins import HeatMapWithTime
import pandas as pd
from watcher import get_logger
from datetime import datetime


# Vamos a loguearlo un poco, que nunca viene mal
logger = get_logger(__name__)

# Starting Flask app
app = Flask(__name__)

# Loading model aimed at severity prediction
# TODO: sacar el entrenamiento de aquí, hacerlo fuera y sólo cagar un fichero .pkl
# Se podría actualizar el modelo cada vez que se divulguen nuevos datos,
# que es de higos a brevas.
logger.info("Starting with RandomForest model loading and training.")
eRF = eng.RandomForest()

basedir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(basedir, 'out_csv')
fpath_val = os.path.join(data_dir, 'geo_out.csv')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-file/", methods=["GET", "POST"])
def upload_image():
    return render_template("upload_file.html")

@app.route("/visualize", methods=["GET", "POST"])
def visualize_map():

    d = request.values['delim']
    dir_c = request.values['direccion']

    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join(basedir, "uploads", secure_filename(f.filename)))
    
    if d == '1':
        delim = ';'
    elif d == '2':
        delim = ','
    else:
        delim = '|'
    
    m = mainProc(secure_filename(f.filename), delim, dir_c)
    global eRF
    eRF.validate(1, fpath_val, folds=10)


    # Para crear el heatmap
    start_coords = (40.416775, -3.703790)
    # Creando el mapa
    folium_map = folium.Map(location=start_coords, 
                            zoom_start=14)
    
    return render_template("visualize.html", prob = '0.0%', les = '0')


@app.route("/get_heatmap", methods=["GET","POST"])
def heatmap():
    # Coordenadas del centro del mapa
    start_coords = (40.416775, -3.703790)
    # Creando el mapa
    folium_map = folium.Map(location=start_coords, 
                            zoom_start=14)

    # Leamos el dataset
    df = pd.read_csv("out_csv/geo_out.csv", delimiter=';')
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
    return folium_map._repr_html_()

@app.route("/get_click", methods=["GET","POST"])
def info_click():
    global eRF
    if request.method == 'POST':
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        #print("Has clickado en: " + str(latitude) + ", " + str(longitude) )
        x_test = np.array([float(latitude), float(longitude)])
        les = eRF.predict_value(x_test.reshape(1, -1))
        print("You have clicked in {}, {} . Predicted Harm level in case of accident: {}".format(latitude, longitude, les))
        logger.info("You have clicked in {}, {} . Predicted Harm level in case of accident: {}".format(latitude, longitude, les))
        #print("Lesividad : {}".format(les))
        p = dict()
        p['lesividad'] = les
    return jsonify(p)


if __name__ == '__main__':
    logger.info("Starting app at {}".format(datetime.now()))
    app.run(debug=True)