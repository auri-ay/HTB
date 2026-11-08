
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from bbox_annotation import Annotation, BoundingBox
from tool import crop_rgb

Image.MAX_IMAGE_PIXELS = None    # the logs are very tall

# Black marks
BLACK_CUTOFF = 40 * 3            # a pixel is black if r + g + b is below this
NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MAX_STDS = 2                     # ignore values more than this many standard deviations from a window's mean
GREY_SYMBOL_SHARE = 0.4          # if black is less than this share of a column's dark pixels, its symbols are drawn in grey...
GREY_INK_BRIGHTNESS = 0.75       # ...and any pixel darker than this fraction of its row's background brightness is a symbol

# Sliding window
WINDOW_FEET = 5                  # window height in feet
MIN_WINDOW_PX = 40               # ...but at least this many pixels, so each window holds a few symbols
WINDOW_STEPS = 4                 # the window moves down a quarter of its height each step

# Changes in the black marks
CHANGE_THRESHOLD = 0.4           # a jump bigger than this fraction of a graph's usual range counts as a big change
WOBBLE_FACTOR = 5                # ...but only if it is this many times bigger than the graph's usual wobble
MEASURES = ["Mean Size", "Mean X Change", "Mean Y Change"]

# Changes in background colour
COLOUR_CHANGE = 40               # a row this far from the current band's colour starts a new band (white to black is 441)
MIN_BAND_FEET = 1.5              # ...but only if the new colour holds for this many feet
MIN_BAND_PX = 10                 # ...and at least this many rows (ignores thin lines and text)
BAND_SHARE = 0.8                 # ...in at least this share of those rows (allows for thin stripes, text and symbols)

# Depth scale
SLICE_HEIGHT = 5000              # read the depth column in slices this many pixels tall
MIN_READINGS = 5                 # depth readings that must agree on one straight line


@dataclass
class ColumnAnalysis:
    boxes: dict                                  # depth_column, interpreted_lithology_column, lithology_legend
    pixels_per_foot: float
    row_colours: np.ndarray = field(repr=False)  # one background colour per column row
    islands: dict = field(repr=False)            # every island's size, x / y change and centre row, top to bottom
    change_ys: list[int] = field(default_factory=list)   # change lines, in image rows


# ---------------------------------------------------------------- depth scale

def read_depth_numbers(image: Image.Image, depth_box: BoundingBox, slice_top: int, slice_bottom: int, work_dir: str) -> list[tuple[float, int]]:
    """(image row, depth) for each depth number tesseract can read in one slice of the depth column.
    The numbers may be upright or printed sideways, so each way round is tried and the best kept."""
    depth_slice = crop_rgb(image, (depth_box[0], slice_top, depth_box[2], slice_bottom)).convert("L")
    zoom = 3 if depth_slice.width < 120 else 2 if depth_slice.width < 250 else 1
    depth_slice = depth_slice.resize((depth_slice.width * zoom, depth_slice.height * zoom))
    slice_path = str(Path(work_dir) / "depth_slice.png")

    best = []
    for turn in (0, 90, 270):
        depth_slice.rotate(turn, expand=True).save(slice_path)
        try:
            ocr = subprocess.run(["tesseract", slice_path, "stdout", "--psm", "11", "-c", "tessedit_char_whitelist=0123456789,", "tsv"],
                                 capture_output=True, text=True).stdout
        except FileNotFoundError:            # tesseract isn't installed
            return []

        readings = []
        for line in ocr.splitlines()[1:]:
            parts = line.split("\t")
            if len(parts) < 12 or float(parts[10]) < 50:          # skip low-confidence words
                continue
            text = parts[11].replace(",", "")
            if not re.fullmatch(r"\d{2,5}", text):
                continue
            left, top, word_width, word_height = (int(value) for value in parts[6:10])
            if turn == 0:
                y_in_slice = top + word_height / 2
            elif turn == 90:                                       # turned anticlockwise: row = x
                y_in_slice = left + word_width / 2
            else:                                                  # turned clockwise: row = height - x
                y_in_slice = depth_slice.height - (left + word_width / 2)
            readings.append((slice_top + y_in_slice / zoom, int(text)))
        if len(readings) > len(best):
            best = readings
    return best


def agreeing_readings(readings: list[tuple[float, int]]) -> list[tuple[float, int]]:
    """The largest set of readings that lie on one straight line (image row against depth), within 2 ft."""
    best = []
    for i in range(len(readings)):
        for j in range(i + 1, min(i + 8, len(readings))):
            row_gap = readings[j][0] - readings[i][0]
            depth_gap = readings[j][1] - readings[i][1]
            if row_gap <= 20 or depth_gap <= 0 or not 0.5 <= row_gap / depth_gap <= 60:
                continue
            scale = row_gap / depth_gap
            agreeing = [(y, d) for y, d in readings if abs((y - readings[i][0]) / scale - (d - readings[i][1])) <= 2]
            if len(agreeing) > len(best):
                best = agreeing
    return best


def depth_scale(image: Image.Image, depth_box: BoundingBox, name: str) -> float:
    """Pixels per foot, from the depth numbers in the depth column.
    If too few can be read, use the file name: a depth range (e.g. "3346ft-13925ft") or a scale (e.g. "5in").
    Failing that, assume a 5-inch log."""
    readings = []
    best = []
    slice_tops = np.linspace(depth_box[1], max(depth_box[3] - SLICE_HEIGHT, depth_box[1]), 8).astype(int)
    with tempfile.TemporaryDirectory() as work_dir:
        for slice_top in slice_tops:                 # slices spread down the column, until plenty agree
            slice_bottom = min(int(slice_top) + SLICE_HEIGHT, depth_box[3])
            readings = sorted(set(readings + read_depth_numbers(image, depth_box, int(slice_top), slice_bottom, work_dir)))
            best = agreeing_readings(readings)
            if len(best) >= 3 * MIN_READINGS:
                break

    if len(best) >= MIN_READINGS:
        return float(np.polyfit([d for _, d in best], [y for y, _ in best], 1)[0])

    dpi = float(image.info.get("dpi", (200, 200))[0])
    depth_range = re.search(r"(\d{3,5})\s*(?:ft)?\s*-\s*(\d{3,5})\s*ft", name, re.I)
    scale_in_name = re.search(r"(\d)\s*-?\s*in", name, re.I) or re.search(r"(?:md|tvd)(\d)(?!\d)", name, re.I)
    if depth_range:
        return (depth_box[3] - depth_box[1]) / (int(depth_range.group(2)) - int(depth_range.group(1)))
    if scale_in_name:
        return dpi * int(scale_in_name.group(1)) / 100
    return dpi * 5 / 100


# ---------------------------------------------------------------- measuring the column

def row_colours(column: Image.Image) -> np.ndarray:
    """Each row's background colour: the median colour of that row's pixels that aren't black
    (the median ignores thin coloured symbols). Rows that are entirely black come out black."""
    pixels = np.array(column).astype(np.float32)
    not_black = pixels.sum(axis=2) >= BLACK_CUTOFF
    pixels[~not_black] = np.nan                       # leave black pixels out
    pixels[~not_black.any(axis=1), 0] = 0             # rows that are entirely black stay black
    return np.nanmedian(pixels, axis=1)


def ink_mask(column: Image.Image, colours: np.ndarray | None = None) -> np.ndarray:
    """The pixels that are part of a symbol. Normally that's the black pixels. But if black is less than
    GREY_SYMBOL_SHARE of the column's dark pixels, its symbols are drawn in grey or colour, so any pixel
    clearly darker than its row's background counts. colours: the column's row_colours, if already worked out."""
    colours = row_colours(column) if colours is None else colours
    brightness = np.array(column).astype(np.int32).sum(axis=2)
    black = brightness < BLACK_CUTOFF
    darker = brightness < colours.sum(axis=1)[:, None] * GREY_INK_BRIGHTNESS
    if black.sum() >= GREY_SYMBOL_SHARE * (black | darker).sum():
        return black
    return black | darker


def find_islands(column: Image.Image, ink: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Every island (group of touching symbol pixels, normally black) in the column, sorted top to bottom.
    Returns arrays of each island's size (pixels), x change, y change and centre row (within the column).
    ink: the column's ink_mask, if already worked out."""
    black = ink_mask(column) if ink is None else ink
    height, width = black.shape
    unvisited = black.tolist()
    sizes, x_changes, y_changes, centre_ys = [], [], [], []

    rows, cols = np.nonzero(black)                    # black pixels, row by row, left to right
    for y, x in zip(rows.tolist(), cols.tolist()):
        if not unvisited[y][x]:
            continue                                  # already part of an island

        unvisited[y][x] = False
        to_visit = [(x, y)]
        island_size = 0
        min_x, max_x, min_y, max_y = x, x, y, y
        while to_visit:
            px, py = to_visit.pop()
            island_size += 1
            min_x, max_x = min(min_x, px), max(max_x, px)
            min_y, max_y = min(min_y, py), max(max_y, py)
            for dx, dy in NEIGHBOURS:
                nx, ny = px + dx, py + dy
                if 0 <= nx < width and 0 <= ny < height and unvisited[ny][nx]:
                    unvisited[ny][nx] = False
                    to_visit.append((nx, ny))

        sizes.append(island_size)
        x_changes.append(max_x - min_x + 1)
        y_changes.append(max_y - min_y + 1)
        centre_ys.append((min_y + max_y) / 2)

    centre_ys = np.array(centre_ys)
    order = np.argsort(centre_ys)
    return {"sizes": np.array(sizes)[order], "x_changes": np.array(x_changes)[order],
            "y_changes": np.array(y_changes)[order], "centre_ys": centre_ys[order]}


def filtered_mean(values: np.ndarray) -> float:
    """Mean, ignoring values more than MAX_STDS standard deviations from it."""
    return values[abs(values - values.mean()) <= MAX_STDS * values.std()].mean()


def window_features(islands: dict[str, np.ndarray], column_height: int, window_height: int, column_top: int) -> pd.DataFrame:
    """Slide a window down the column and average the islands whose centre is inside it.
    One row per window, with its Bottom Y in image rows."""
    window_tops = np.arange(0, max(column_height - window_height, 0) + 1, window_height / WINDOW_STEPS)
    results = []
    for window_top in window_tops:
        window_bottom = window_top + window_height
        first = np.searchsorted(islands["centre_ys"], window_top)     # first island whose centre is in the window
        last = np.searchsorted(islands["centre_ys"], window_bottom)   # first island below the window
        if last > first:
            mean_size = filtered_mean(islands["sizes"][first:last])
            mean_x_change = filtered_mean(islands["x_changes"][first:last])
            mean_y_change = filtered_mean(islands["y_changes"][first:last])
        else:
            mean_size = mean_x_change = mean_y_change = 0
        results.append({
            "Bottom Y": column_top + round(window_bottom),
            "Mean Size": mean_size,
            "Mean X Change": mean_x_change,
            "Mean Y Change": mean_y_change,
        })
    return pd.DataFrame(results)


# ---------------------------------------------------------------- finding changes

def shape_changes(features: pd.DataFrame) -> list[int]:
    """Rows where the black marks change: compare each window with the next one that doesn't overlap it."""
    change_score = np.zeros(len(features))
    for measure in MEASURES:
        values = features[measure].to_numpy()
        low, high = np.percentile(values, [1, 99])
        if high == low:
            continue
        values = np.clip(values, low, high)                    # stop one huge spike from setting the scale
        jump = abs(values[WINDOW_STEPS:] - values[:-WINDOW_STEPS])
        score = jump / (high - low) / CHANGE_THRESHOLD
        score[jump <= WOBBLE_FACTOR * np.median(jump)] = 0
        change_score[:-WINDOW_STEPS] = np.maximum(change_score[:-WINDOW_STEPS], score)

    # Each run of big changes is one change; put the line where the change is biggest
    change_ys = []
    i = 0
    while i < len(change_score):
        if change_score[i] > 1:
            run_end = i
            while run_end + 1 < len(change_score) and change_score[run_end + 1] > 1:
                run_end += 1
            biggest = i + np.argmax(change_score[i:run_end + 1])
            change_ys.append(int(features["Bottom Y"].iloc[biggest]))    # where the two windows meet
            i = run_end + 1
        else:
            i += 1
    return change_ys


def colour_changes(colours: np.ndarray, band_rows: int, column_top: int) -> list[int]:
    """Rows where a new background colour band starts: a row far from the current band's colour,
    as long as most of the next band_rows rows keep the new colour."""
    change_ys = []
    start = 0
    while start < min(band_rows, len(colours) - 1) and not colours[start].any():
        start += 1                                    # skip the column's black top frame
    band_colour = colours[start]
    y = start + 1
    while y < len(colours):
        if np.linalg.norm(colours[y] - band_colour) > COLOUR_CHANGE:
            next_rows = colours[y:y + band_rows]
            share_matching = np.mean(np.linalg.norm(next_rows - colours[y], axis=1) <= COLOUR_CHANGE)
            if len(next_rows) == band_rows and share_matching >= BAND_SHARE:
                change_ys.append(column_top + y)
                band_colour = colours[y]
                y += band_rows
                continue
        y += 1
    return change_ys


def combine_changes(colour_change_ys: list[int], shape_change_ys: list[int], window_height: int) -> list[int]:
    """Colour lines are exact to the row, so drop a black-mark line that is right next to one."""
    change_ys = list(colour_change_ys)
    for shape_y in shape_change_ys:
        if all(abs(shape_y - colour_y) > window_height / 2 for colour_y in colour_change_ys):
            change_ys.append(shape_y)
    return sorted(change_ys)


# ---------------------------------------------------------------- putting it together

def analyse_image(image: Image.Image, annotation: Annotation) -> ColumnAnalysis:
    """Run the whole method on one image, using its annotation (from bbox_annotation.annotation_from_json)."""
    boxes = dict(annotation["bounding_boxs"])
    column_box = boxes["interpreted_lithology_column"]

    pixels_per_foot = depth_scale(image, boxes["depth_column"], Path(annotation["image_name"]).stem)
    window_height = max(round(WINDOW_FEET * pixels_per_foot), MIN_WINDOW_PX)
    band_rows = max(round(MIN_BAND_FEET * pixels_per_foot), MIN_BAND_PX)

    column = crop_rgb(image, column_box)
    colours = row_colours(column)
    islands = find_islands(column, ink_mask(column, colours))
    features = window_features(islands, column.height, window_height, column_box[1])

    change_ys = combine_changes(colour_changes(colours, band_rows, column_box[1]), shape_changes(features), window_height)
    return ColumnAnalysis(boxes=boxes, pixels_per_foot=pixels_per_foot, row_colours=colours,
                          islands=islands, change_ys=change_ys)


def interval_stats(image: Image.Image, annotation: Annotation) -> pd.DataFrame:
    """One row per stretch of column between change lines: where it starts and ends (image rows), then the
    measurements averaged over that stretch:
      Island Count, Islands Per Foot            - islands centred in the stretch, in total and per foot of depth
      Mean Size, Mean X Change, Mean Y Change   - of those islands, ignoring values more than MAX_STDS standard
                                                  deviations from the mean (0 if there are no islands)
      Background R, Background G, Background B  - the average of the stretch's row colours
    annotation: the image's annotation, from bbox_annotation.annotation_from_json."""
    analysis = analyse_image(image, annotation)
    column_box = analysis.boxes["interpreted_lithology_column"]
    column_top = column_box[1]
    islands = analysis.islands
    edges = [column_box[1]] + [y for y in analysis.change_ys if column_box[1] < y < column_box[3]] + [column_box[3]]

    rows = []
    for start_y, end_y in zip(edges[:-1], edges[1:]):
        if end_y <= start_y:
            continue
        first = np.searchsorted(islands["centre_ys"], start_y - column_top)    # first island centred in the stretch
        last = np.searchsorted(islands["centre_ys"], end_y - column_top)       # first island below it
        if last > first:
            mean_size = filtered_mean(islands["sizes"][first:last])
            mean_x_change = filtered_mean(islands["x_changes"][first:last])
            mean_y_change = filtered_mean(islands["y_changes"][first:last])
        else:
            mean_size = mean_x_change = mean_y_change = 0
        background = analysis.row_colours[start_y - column_top:end_y - column_top].mean(axis=0)
        feet = (end_y - start_y) / analysis.pixels_per_foot

        rows.append({
            "Start Y": start_y,
            "End Y": end_y,
            "Island Count": int(last - first),
            "Islands Per Foot": (last - first) / feet,
            "Mean Size": mean_size,
            "Mean X Change": mean_x_change,
            "Mean Y Change": mean_y_change,
            "Background R": background[0],
            "Background G": background[1],
            "Background B": background[2],
        })
    return pd.DataFrame(rows)
