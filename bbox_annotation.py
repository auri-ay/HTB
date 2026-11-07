import pathlib
from typing import TypedDict, Any

class BoundingBox(TypedDict):
    depth_column: list[int]
    interpreted_lithology_column: list[int]
    lithology_legend: list[int]

class Annotation(TypedDict):
    image_name: str
    width: int
    height: int
    bounding_boxs: BoundingBox

def annotation_from_json(json: dict[str, Any]) -> Annotation:
    return Annotation(image_name=json["image_name"], width=json["width"], height=json["height"], bounding_boxs=BoundingBox(**json["bounding_boxes"]))