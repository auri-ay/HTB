import base64
import json
import tempfile
from io import BytesIO
from typing import Any

import flask
from PIL import Image
from flask import Flask, request
import pickle

from bbox_annotation import TableRow
from data import post_body_from_json
import pandas as pd

from thing import table_row_from_json
from tool import crop_rgb

app = Flask(__name__)


def magic_function(thing: list[dict[str, Any]]) -> list[TableRow]:
    return list(map(table_row_from_json, thing))


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

    img2 = Image.open(body["image"])
    croped2 = crop_rgb(img2, body["annotation"]["bounding_boxs"]["lithology_legend"])
    buffered2 = BytesIO()
    croped2.save(buffered2, format="JPEG")
    legend_image = base64.b64encode(buffered2.getvalue()).decode('ascii')

    body["image"].close()

    col = body["annotation"]["bounding_boxs"]["interpreted_lithology_column"]
    real_height = abs(col.maxY - col.minY)

    return json.dumps({"start_y": body["annotation"]["bounding_boxs"]["interpreted_lithology_column"].minY,
                       "real_height": real_height, "image": str(img_str), "legend_image": str(legend_image), "result": magic_function(json.loads(
            '[{"top_depth":25400,"bottom_depth":25420,"lithology":"Shale","top_y":4409,"bottom_y":4809,"cluster_id":0},{"top_depth":25420,"bottom_depth":25430,"lithology":"Siltstone","top_y":4809,"bottom_y":5010,"cluster_id":1},{"top_depth":25430,"bottom_depth":25490,"lithology":"Shale","top_y":5010,"bottom_y":6211,"cluster_id":0},{"top_depth":25490,"bottom_depth":25500,"lithology":"Siltstone","top_y":6211,"bottom_y":6411,"cluster_id":1},{"top_depth":25500,"bottom_depth":25510,"lithology":"Shale","top_y":6411,"bottom_y":6611,"cluster_id":0},{"top_depth":25510,"bottom_depth":25520,"lithology":"Sandstone","top_y":6611,"bottom_y":6811,"cluster_id":2},{"top_depth":25520,"bottom_depth":25530,"lithology":"Siltstone","top_y":6811,"bottom_y":7012,"cluster_id":1},{"top_depth":25530,"bottom_depth":25540,"lithology":"Sandstone","top_y":7012,"bottom_y":7212,"cluster_id":2},{"top_depth":25540,"bottom_depth":25550,"lithology":"Siltstone","top_y":7212,"bottom_y":7412,"cluster_id":1},{"top_depth":25550,"bottom_depth":25560,"lithology":"Shale","top_y":7412,"bottom_y":7612,"cluster_id":0},{"top_depth":25560,"bottom_depth":25570,"lithology":"Sandstone","top_y":7612,"bottom_y":7812,"cluster_id":2},{"top_depth":25570,"bottom_depth":25580,"lithology":"Shale","top_y":7812,"bottom_y":8013,"cluster_id":0},{"top_depth":25580,"bottom_depth":25590,"lithology":"Siltstone","top_y":8013,"bottom_y":8213,"cluster_id":1},{"top_depth":25590,"bottom_depth":25630,"lithology":"Shale","top_y":8213,"bottom_y":9014,"cluster_id":0},{"top_depth":25630,"bottom_depth":25640,"lithology":"Sandstone","top_y":9014,"bottom_y":9214,"cluster_id":2},{"top_depth":25640,"bottom_depth":25690,"lithology":"Shale","top_y":9214,"bottom_y":10215,"cluster_id":0},{"top_depth":25690,"bottom_depth":25700,"lithology":"Siltstone","top_y":10215,"bottom_y":10415,"cluster_id":1}]'))})


@app.route('/api/convert', methods=['POST'])
def convert():
    body = request.get_data(as_text=True)
    with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp:
        tmp.write(base64.b64decode(body.split(",", maxsplit=1)[1]))
        img = Image.open(tmp.name)
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('ascii')
        return img_str
