from pathlib import Path
from typing import TypedDict, NamedTuple

import pandas as pd
from PIL import Image

from bbox_annotation import Annotation, Box


class TableRow(TypedDict):
    name: str
    image: Path
    width: int
    height: int
    depth_column: Box
    interpreted_lithology_column: Box
    lithology_legend: Box


class SizeMismatch(Exception):
    pass


def thing(image_path: Path, annotation: Annotation) -> TableRow:
    # if image.width != annotation["width"] or image.height != annotation["height"]:
    #     SizeMismatch (
    #         f"Annotation has w: {annotation["width"]} h: {annotation["height"]}, but image has dimension w: {image.width} h: {image.height}")

    return TableRow(
        name=image_path.stem, image=image_path, width=annotation["width"], height=annotation["height"],
        depth_column=annotation["bounding_boxs"]["depth_column"],
        interpreted_lithology_column=annotation["bounding_boxs"]["interpreted_lithology_column"],
        lithology_legend=annotation["bounding_boxs"]["lithology_legend"]
    )
