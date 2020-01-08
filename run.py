from flask import Flask, render_template, request, redirect, jsonify
from werkzeug import secure_filename
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import numpy as np
import folium
from folium.plugins import HeatMap, HeatMapWithTime
import pandas as pd
from watcher import get_logger
from datetime import datetime, date
from data_handler import check_updates, load_model, load_KDEmodel
from engine import Engine as eng
from reverse_coordinates import Reverse

# Vamos a loguearlo un poco, que nunca viene mal
logger = get_logger(__name__)

# Starting Flask app
app = Flask(__name__)

# Loading model aimed at severity prediction
# TODO: sacar el entrenamiento de aquí, hacerlo fuera y sólo cagar un fichero .pkl
# Se podría actualizar el modelo cada vez que se divulguen nuevos datos,
# que es de higos a brevas.


basedir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(basedir, 'out_csv')
fpath_val = os.path.join(data_dir, 'geo_out.csv')
model = None
kde_model = None
reverse = None

@app.route("/")
def index():
    # Aqui se llevara el control sobre la geolocalizacion de direcciones y sobre el entrenamiento del modelo
    check_updates()
    return render_template("index.html")

@app.route("/visualize", methods=["GET", "POST"])
def visualize_map():

    global model
    model = load_model()

    global kde_model
    kde_model = load_KDEmodel()

    global reverse
    reverse = Reverse()

    return render_template("visualize.html", prob = '0.0%', les = '0')

@app.route("/get_click", methods=["GET","POST"])
def info_click():
    if request.method == 'POST':
        latitude = request.form['latitude']
        longitude = request.form['longitude']

        x_test = np.array([float(latitude), float(longitude)])

        global model
        les = int(model.predict(x_test.reshape(1,-1))[0])

        global kde_model
        prob = kde_model.predict_value(x_test.reshape(1,-1))

        global reverse
        addr = reverse.reverse_coord(latitude, longitude)

        print("You have clicked in {}, {} . Predicted Harm level in case of accident: {}".format(latitude, longitude, les))
        logger.info("You have clicked in {}, {} . Predicted Harm level in case of accident: {}".format(latitude, longitude, les))
        print('Has hecho click en {}'.format(addr))
        
        p = dict()
        p['lesividad'] = les
        p['prob'] = prob[0]
        p['address'] = addr

    return jsonify(p)

if __name__ == '__main__':
    logger.info("Starting app at {}".format(datetime.now()))
    app.run(debug=True)