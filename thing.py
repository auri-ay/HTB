from pathlib import Path
from typing import TypedDict, NamedTuple, Any

import pandas as pd
from PIL import Image

from bbox_annotation import Annotation, BoundingBox


class TableRow(TypedDict):
    top_depth: int
    bottom_depth: int
    lithology: str
    top_y: int
    bottom_y: int
    cluster_id: int

def table_row_from_json(dict: dict[str, Any]) -> TableRow:
    return TableRow(top_depth=dict["top_depth"], bottom_depth=dict["bottom_depth"], lithology=dict["lithology"],
                    top_y=dict["top_y"], bottom_y=dict["bottom_y"], cluster_id=dict["cluster_id"])


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
