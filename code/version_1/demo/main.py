"""
    Flask app to host the computed results for better visualization.
"""

from flask import Flask, redirect, url_for, request, render_template, send_file
import os
from glob import glob

app = Flask(__name__)

layer_wise_images_path = "/Users/Janjua/Desktop/QCRI/Work/tcav-nlp/code/version_1/demo/static/images/layer_wise/"
concept_wise_images_path = "/Users/Janjua/Desktop/QCRI/Work/tcav-nlp/code/version_1/demo/static/images/concept_wise/"

def load_the_images_layer_wise():
    layer_images = {}
    for img in glob(layer_wise_images_path + "*.png"):
        layer = img.split('/')[-1].split('.')[0]
        img_p = '/'.join(img.split('/')[-4:])
        layer_images[layer] = "/" + img_p
    return layer_images

def load_the_images_concept_wise():
    concept_images = {}
    for img in glob(concept_wise_images_path + "*.png"):
        concept = img.split('/')[-1].split('.')[0]
        img_p = '/'.join(img.split('/')[-4:])
        concept_images[concept] = "/" + img_p
    return concept_images

@app.route('/randomImg', methods=['GET', 'POST'])
def display_the_layer_wise():
    if request.method == 'POST':
        layer_images = load_the_images_layer_wise()
        val = request.form.get("picker")
        ranimage = layer_images[val]
        return render_template("index.html", ranimage=ranimage)
    return None

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)