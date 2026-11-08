import tempfile
from pathlib import Path
from typing import Callable

from PIL import ImageOps
from PIL import Image
from paddleocr import PaddleOCR

from bbox_annotation import Annotation

from tool import crop_rgb

Image.MAX_IMAGE_PIXELS = None

ocr = PaddleOCR(

use_doc_orientation_classify=False,

use_doc_unwarping=False,

use_textline_orientation=False,

engine="paddle",
)

def get_height(image_path: Path, annotation: Annotation) -> Callable[[int], float]:
    im = crop_rgb(Image.open(image_path), annotation["bounding_boxs"]["depth_column"]).rotate(90, expand=True)
    grayscale = ImageOps.grayscale(im)
    with tempfile.NamedTemporaryFile(suffix=".png") as f:
        grayscale.save(f, format="PNG")
        path = f.name
        results = ocr.predict(path)
        for r in results:
            r.save_to_img("output")
            res = r["res"] if "res" in r else r
            data = list(zip(res["rec_texts"], res["rec_boxes"]))

            for index, (text, box) in enumerate(data):
                width = box[2] - box[0]
                center = box[0] + width / 2

                data.append((int(text), center))

    data.sort(key=lambda x: x[1])

    diff = data[-1][0] - data[0][0]
    width = data[-1][1] - data[0][1]

    return lambda y: data[0][1] + (y - data[0][1]) * (diff / width)