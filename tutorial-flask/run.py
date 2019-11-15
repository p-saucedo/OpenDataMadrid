from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
import os
app = Flask(__name__)
posts = []


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload-file/", methods=["GET", "POST"])
def upload_image():
    return render_template("upload_file.html")

@app.route("/visualize/", methods=["GET", "POST"])
def visualize_map():
    if request.method == "POST":

        if request.files:

            file = request.files["file"]

            print(file)


    return render_template("upload_file.html")

@app.route("/p/<string:slug>/")
def show_post(slug):
    return render_template("post_view.html", slug_title=slug)

@app.route("/admin/post/")
@app.route("/admin/post/<int:post_id>/")
def post_form(post_id=None):
    return render_template("admin/post_form.html", post_id=post_id)

@app.route("/signup/", methods=["GET", "POST"])
def show_signup_form():
    return render_template("signup_form.html")

if __name__ == '__main__':
    app.run(debug=True)