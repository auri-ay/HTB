import base64
import json
from io import BytesIO

import flask
from PIL import Image
from flask import Flask, request
import pickle
from data import post_body_from_json
import pandas as pd

from tool import crop_rgb

app = Flask(__name__)

@app.route('/')
def index():
    return flask.send_file("web/index.html")
@app.route('/<path:path>')
def serve(path):
    return flask.send_from_directory("web/", path)

@app.route('/api/thing', methods=['POST'])
def do_thing():
    # print(request.get_json())
    body = post_body_from_json(request.get_json())
    img = Image.open(body["image"])
    croped = crop_rgb(img, body["annotation"]["bounding_boxs"]["interpreted_lithology_column"])
    buffered = BytesIO()
    croped.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
    body["image"].close()

    with open("dataset.pickle", "rb") as p:
        dataset = pickle.load(p)
        dataset

    return json.dumps({"image": str(img_str), "result": None})