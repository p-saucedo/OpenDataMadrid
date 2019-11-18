from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from werkzeug import secure_filename
from mainProc import mainProc
import os
app = Flask(__name__)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-file/", methods=["GET", "POST"])
def upload_image():
    return render_template("upload_file.html")

@app.route("/visualize", methods=["GET", "POST"])
def visualize_map():
    if request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join("uploads/", secure_filename(f.filename)))
    
    m = mainProc(secure_filename(f.filename), ';')

    return render_template("visualize.html")

@app.route("/get_click", methods=["GET","POST"])
def info_click():
    if request.method == 'POST':
        latitude = request.form['latitude']
        longitude = request.form['longitude']
        print("Has clickado en: " + str(latitude) + ", " + str(longitude) )
    return render_template("visualize.html")

if __name__ == '__main__':
    app.run(debug=True)