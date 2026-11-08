import pathlib
from typing import TypedDict, Any, NamedTuple


class Box(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class BoundingBox(TypedDict):
    depth_column: Box
    interpreted_lithology_column: Box
    lithology_legend: Box


class Annotation(TypedDict):
    image_name: str
    width: int
    height: int
    bounding_boxs: BoundingBox


class TableRow(TypedDict):
    image_name: str
    width: int
    height: int


def bounding_box_from_json(json: dict[str, Any]) -> BoundingBox:
    return BoundingBox(depth_column=Box(*json["depth_column"]),
                       interpreted_lithology_column=Box(*json["interpreted_lithology_column"]),
                       lithology_legend=Box(*json["lithology_legend"]))


def annotation_from_json(json: dict[str, Any]) -> Annotation:
    return Annotation(image_name=json["image_name"], width=json["width"], height=json["height"],
                      bounding_boxs=bounding_box_from_json(json["bounding_boxes"]))
