import base64
import os
import tempfile
from typing import TypedDict, Any

import json

from bbox_annotation import Annotation, annotation_from_json


class PostBody(TypedDict):
    annotation: Annotation
    image: tempfile. _TemporaryFileWrapper

def post_body_from_json(json_body: dict[str, Any])->PostBody:
    imgdata:str = json_body["image"]
    img = base64.b64decode(imgdata.split(",", maxsplit=1)[1])
    f = tempfile.NamedTemporaryFile()
    f.write(img)
    return PostBody(annotation=annotation_from_json(json.loads(json_body["annotation"])), image=f)