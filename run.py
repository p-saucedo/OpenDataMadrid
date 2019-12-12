from flask import Flask, render_template, request, redirect, jsonify
from werkzeug import secure_filename
from mainProc import mainProc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from engine import Engine as eng
import os
import numpy as np


app = Flask(__name__)

eRF = eng.RandomForest()


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
        f.save(os.path.join("uploads/", secure_filename(f.filename)))
    
    if d == '1':
        delim = ';'
    elif d == '2':
        delim = ','
    else:
        delim = '|'
    
    m = mainProc(secure_filename(f.filename), delim, dir_c)
    global eRF
    eRF.validate(1, 'out_csv/geo_out.csv', folds=10)
    
    return render_template("visualize.html", prob = '0.0%', les = '0')

@app.route("/get_click", methods=["GET","POST"])
def info_click():
    global eRF
    if request.method == 'POST':
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        print("Has clickado en: " + str(latitude) + ", " + str(longitude) )
        x_test = np.array([float(latitude), float(longitude)])
        les = eRF.predict_value(x_test.reshape(1, -1))
        print("Lesividad : {}".format(les))
        p = dict()
        p['lesividad'] = les
    return jsonify(p)


if __name__ == '__main__':
 
    app.run(debug=True)