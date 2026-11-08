import pathlib
from typing import TypedDict, Any, NamedTuple


class BoundingBox(NamedTuple):
    minX: int
    minY: int
    maxX: int
    maxY: int
    @property
    def w(self): return self.maxX - self.minX
    @property
    def h(self): return self.maxY - self.minY
    @property
    def cy(self): return (self.minY + self.maxY) / 2

class BoundingBoxes(TypedDict):
    depth_column: BoundingBox
    interpreted_lithology_column: BoundingBox
    lithology_legend: BoundingBox


class Annotation(TypedDict):
    image_name: str
    width: int
    height: int
    bounding_boxs: BoundingBoxes


class TableRow(TypedDict):
    image_name: str
    width: int
    height: int


def bounding_box_from_json(json: dict[str, Any]) -> BoundingBoxes:
    return BoundingBoxes(depth_column=BoundingBox(*json["depth_column"]),
                         interpreted_lithology_column=BoundingBox(*json["interpreted_lithology_column"]),
                         lithology_legend=BoundingBox(*json["lithology_legend"]))


def annotation_from_json(json: dict[str, Any]) -> Annotation:
    return Annotation(image_name=json["image_name"], width=json["width"], height=json["height"],
                      bounding_boxs=bounding_box_from_json(json["bounding_boxes"]))
