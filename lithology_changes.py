"""Find where the interpreted lithology column of a mud-log image changes.

Pass an image path to `analyse_image` and it will:
  1. work out the depth scale (pixels per foot) from the numbers printed in the depth column,
  2. crop the interpreted lithology column,
  3. find every "island" (group of touching black pixels) in the column,
  4. slide a window down the column and average the islands' size and x / y change in each window,
  5. record each row's background colour,
  6. find the rows where the column changes: big jumps in the black marks, or a new colour band.

`plot_analysis` draws the column, its background colour, the three measurements and the change lines.
`intervals` turns the change lines into stretches of column with their depths.

This is the same method as col_to_feat.ipynb, with the same settings.
Reading the depth numbers needs tesseract (`brew install tesseract`); without it the scale comes from the file name.

`plot_all` does every image in a data folder and shows all the graphs.
`interval_stats` gives one row per stretch between change lines, with its start / end row and the stats averaged over it.

Example:
    from lithology_changes import analyse_image, plot_analysis, intervals, plot_all, interval_stats

    interval_stats("dataset/images/0000000009_d-582_0000020435-0001_log06_fe_5in_md.tif")   # a DataFrame

    plot_all("dataset")                          # graphs for every image in dataset/images

    analysis = analyse_image("dataset/images/0000000009_d-582_0000020435-0001_log06_fe_5in_md.tif")
    print(analysis.change_ys)
    plot_analysis(analysis)                      # whole column
    plot_analysis(analysis, top=4400, bottom=7000)   # a close-up, in image rows
    intervals(analysis)

From the terminal:
    python lithology_changes.py dataset                  # show the graphs for every image
    python lithology_changes.py dataset --save graphs    # ...and save them as PNGs in graphs/
    python lithology_changes.py dataset/images/<image>.tif
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from bbox_annotation import Box, annotation_from_json
from tool import crop_rgb

Image.MAX_IMAGE_PIXELS = None    # the logs are very tall

# Black marks
BLACK_CUTOFF = 40 * 3            # a pixel is black if r + g + b is below this
NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MAX_STDS = 2                     # ignore values more than this many standard deviations from a window's mean

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

# Graphs for a whole data folder
LARGE_IMAGE = 20000              # columns taller than this (px) also get a close-up of their top
CLOSE_UP_HEIGHT = 5000           # how many px the close-up shows
IMAGE_TYPES = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


@dataclass
class DepthScale:
    pixels_per_foot: float
    source: str                                  # where the scale came from
    fit: tuple[float, float] | None = None       # row = fit[0] * depth + fit[1], when read from the depth numbers

    def depth_at(self, row: float) -> float | None:
        """The depth (ft) at an image row, if the depth numbers could be read."""
        if self.fit is None:
            return None
        return (row - self.fit[1]) / self.fit[0]


@dataclass
class ColumnAnalysis:
    name: str
    image_path: Path
    boxes: dict                                  # depth_column, interpreted_lithology_column, lithology_legend
    scale: DepthScale
    window_height: int                           # pixels
    band_rows: int                               # pixels
    column: Image.Image = field(repr=False)      # the cropped lithology column (RGB)
    row_colours: np.ndarray = field(repr=False)  # one background colour per column row
    features: pd.DataFrame = field(repr=False)   # one row per window
    islands: dict = field(repr=False)            # every island's size, x / y change and centre row, top to bottom
    change_ys: list[int] = field(default_factory=list)          # all change lines, in image rows
    shape_change_ys: list[int] = field(default_factory=list)    # from the black marks
    colour_change_ys: list[int] = field(default_factory=list)   # from the background colour


def load_boxes(image_path: str | Path, annotation_dir: str | Path = "dataset/bbox_annotation") -> dict:
    """The bounding boxes for an image, from the annotation JSON with the same name."""
    annotation_path = Path(annotation_dir) / f"{Path(image_path).stem}.json"
    with open(annotation_path) as f:
        annotation = annotation_from_json(json.load(f))
    return dict(annotation["bounding_boxs"])


# ---------------------------------------------------------------- depth scale

def read_depth_numbers(image: Image.Image, depth_box: Box, slice_top: int, slice_bottom: int, work_dir: str) -> list[tuple[float, int]]:
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


def depth_scale(image: Image.Image, depth_box: Box, name: str) -> DepthScale:
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
        slope, intercept = np.polyfit([d for _, d in best], [y for y, _ in best], 1)
        return DepthScale(float(slope), f"depth numbers ({len(best)} readings agree)", (float(slope), float(intercept)))

    dpi = float(image.info.get("dpi", (200, 200))[0])
    depth_range = re.search(r"(\d{3,5})\s*(?:ft)?\s*-\s*(\d{3,5})\s*ft", name, re.I)
    scale_in_name = re.search(r"(\d)\s*-?\s*in", name, re.I) or re.search(r"(?:md|tvd)(\d)(?!\d)", name, re.I)
    if depth_range:
        feet = int(depth_range.group(2)) - int(depth_range.group(1))
        return DepthScale((depth_box[3] - depth_box[1]) / feet, f"depth range in file name ({depth_range.group(0)})")
    if scale_in_name:
        return DepthScale(dpi * int(scale_in_name.group(1)) / 100, f"scale in file name at {dpi:.0f} dpi")
    return DepthScale(dpi * 5 / 100, f"assumed 5-inch log at {dpi:.0f} dpi")


# ---------------------------------------------------------------- measuring the column

def row_colours(column: Image.Image) -> np.ndarray:
    """Each row's background colour: the median colour of that row's pixels that aren't black
    (the median ignores thin coloured symbols). Rows that are entirely black come out black."""
    pixels = np.array(column).astype(np.float32)
    not_black = pixels.sum(axis=2) >= BLACK_CUTOFF
    pixels[~not_black] = np.nan                       # leave black pixels out
    pixels[~not_black.any(axis=1), 0] = 0             # rows that are entirely black stay black
    return np.nanmedian(pixels, axis=1)


def find_islands(column: Image.Image) -> dict[str, np.ndarray]:
    """Every island (group of touching black pixels) in the column, sorted top to bottom.
    Returns arrays of each island's size (pixels), x change, y change and centre row (within the column)."""
    black = np.array(column).sum(axis=2) < BLACK_CUTOFF
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
    One row per window, with Top Y / Bottom Y in image rows."""
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
            "Top Y": column_top + round(window_top),
            "Bottom Y": column_top + round(window_bottom),
            "Islands": int(last - first),
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
    band_colour = colours[0]
    y = 1
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

def analyse_image(image_path: str | Path, boxes: dict | None = None,
                  annotation_dir: str | Path = "dataset/bbox_annotation") -> ColumnAnalysis:
    """Run the whole method on one image.
    boxes: the image's bounding boxes (depth_column, interpreted_lithology_column, ...); if not given,
    they are read from the annotation JSON with the same name in annotation_dir."""
    image_path = Path(image_path)
    name = image_path.stem
    boxes = boxes if boxes is not None else load_boxes(image_path, annotation_dir)
    column_box = boxes["interpreted_lithology_column"]

    image = Image.open(image_path)
    scale = depth_scale(image, boxes["depth_column"], name)
    window_height = max(round(WINDOW_FEET * scale.pixels_per_foot), MIN_WINDOW_PX)
    band_rows = max(round(MIN_BAND_FEET * scale.pixels_per_foot), MIN_BAND_PX)

    column = crop_rgb(image, column_box)
    colours = row_colours(column)
    islands = find_islands(column)
    features = window_features(islands, column.height, window_height, column_box[1])

    shape_change_ys = shape_changes(features)
    colour_change_ys = colour_changes(colours, band_rows, column_box[1])
    change_ys = combine_changes(colour_change_ys, shape_change_ys, window_height)

    return ColumnAnalysis(name=name, image_path=image_path, boxes=boxes, scale=scale,
                          window_height=window_height, band_rows=band_rows, column=column,
                          row_colours=colours, features=features, islands=islands, change_ys=change_ys,
                          shape_change_ys=shape_change_ys, colour_change_ys=colour_change_ys)


def intervals(analysis: ColumnAnalysis) -> pd.DataFrame:
    """The stretches of column between change lines, with their image rows and (if known) depths."""
    column_box = analysis.boxes["interpreted_lithology_column"]
    edges = [column_box[1]] + [y for y in analysis.change_ys if column_box[1] < y < column_box[3]] + [column_box[3]]
    rows = []
    for top_y, bottom_y in zip(edges[:-1], edges[1:]):
        rows.append({
            "top_y": top_y,
            "bottom_y": bottom_y,
            "top_depth": analysis.scale.depth_at(top_y),
            "bottom_depth": analysis.scale.depth_at(bottom_y),
        })
    return pd.DataFrame(rows)


def interval_stats(image: str | Path | ColumnAnalysis, boxes: dict | None = None,
                   annotation_dir: str | Path = "dataset/bbox_annotation") -> pd.DataFrame:
    """One row per stretch of column between change lines: where it starts and ends (image rows), then the
    measurements averaged over that stretch:
      Island Count, Islands Per Foot            - islands centred in the stretch, in total and per foot of depth
      Mean Size, Mean X Change, Mean Y Change   - of those islands, ignoring values more than MAX_STDS standard
                                                  deviations from the mean (0 if there are no islands)
      Background R, Background G, Background B  - the average of the stretch's row colours
    image: an image path (it is analysed first) or a ColumnAnalysis you already have."""
    analysis = image if isinstance(image, ColumnAnalysis) else analyse_image(image, boxes, annotation_dir)
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
        feet = (end_y - start_y) / analysis.scale.pixels_per_foot

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


def plot_analysis(analysis: ColumnAnalysis, top: int | None = None, bottom: int | None = None) -> plt.Figure:
    """The column, its background colour, the three measurements and the change lines (red).
    top / bottom (image rows) zoom in on part of the column; by default the whole column is shown."""
    column_box = analysis.boxes["interpreted_lithology_column"]
    top = column_box[1] if top is None else max(top, column_box[1])
    bottom = column_box[3] if bottom is None else min(bottom, column_box[3])

    column_part = analysis.column.crop((0, top - column_box[1], analysis.column.width, bottom - column_box[1]))
    if column_part.height > 4000:                     # shrink very tall stretches so plotting stays fast
        column_part = column_part.resize((column_part.width, 4000))
    colours_part = analysis.row_colours[top - column_box[1]:bottom - column_box[1]]
    features = analysis.features
    features_part = features[(features["Top Y"] >= top) & (features["Top Y"] < bottom)]
    changes_part = [y for y in analysis.change_ys if top <= y <= bottom]

    fig, axes = plt.subplots(1, 5, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [1, 1, 3, 3, 3]})
    axes[0].imshow(column_part, extent=[0, 1, bottom, top], aspect="auto")
    axes[0].set_xticks([])
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Top Y (px)")

    axes[1].imshow(colours_part[:, None, :] / 255, extent=[0, 1, bottom, top], aspect="auto")
    axes[1].set_xticks([])
    axes[1].set_xlabel("Background")

    for ax, measure in zip(axes[2:], MEASURES):
        ax.plot(features_part[measure], features_part["Top Y"])
        ax.set_xlabel(f"{measure} (px)")

    for ax in axes:
        for change_y in changes_part:
            ax.axhline(change_y, color="red", linewidth=0.8)

    axes[0].set_ylim(bottom, top)                     # top of the column at the top, like the image
    zoomed = (top, bottom) != (column_box[1], column_box[3])
    where = f" (rows {top}-{bottom})" if zoomed else ""
    fig.suptitle(f"{analysis.name}{where}: {len(changes_part)} big changes")
    return fig


def image_and_annotation_dirs(data_path: str | Path) -> tuple[Path, Path]:
    """Accept either the data folder (with images/ and bbox_annotation/ inside) or its images folder."""
    data_path = Path(data_path)
    if (data_path / "images").is_dir():
        return data_path / "images", data_path / "bbox_annotation"
    return data_path, data_path.parent / "bbox_annotation"


def plot_all(data_path: str | Path = "dataset", close_ups: bool = True, save_dir: str | Path | None = None) -> dict[str, list[int]]:
    """Analyse every image in a data folder and show its graphs: the whole column, plus a full-resolution
    close-up of the top of large images. Images without an annotation are skipped.
    save_dir: also save each graph there as a PNG. Returns each image's change lines (image rows)."""
    image_dir, annotation_dir = image_and_annotation_dirs(data_path)
    image_paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    all_change_ys = {}
    for number, image_path in enumerate(image_paths, start=1):
        if not (annotation_dir / f"{image_path.stem}.json").exists():
            print(f"{number}/{len(image_paths)} {image_path.stem}: no annotation, skipped")
            continue

        analysis = analyse_image(image_path, annotation_dir=annotation_dir)
        all_change_ys[analysis.name] = analysis.change_ys
        print(f"{number}/{len(image_paths)} {analysis.name}: {analysis.scale.pixels_per_foot:.2f} px/ft "
              f"({analysis.scale.source}), {len(analysis.change_ys)} changes")

        column_box = analysis.boxes["interpreted_lithology_column"]
        charts = [("whole column", None, None)]
        if close_ups and column_box[3] - column_box[1] > LARGE_IMAGE:
            charts.append((f"top {CLOSE_UP_HEIGHT} px", column_box[1], column_box[1] + CLOSE_UP_HEIGHT))

        for chart_name, top, bottom in charts:
            fig = plot_analysis(analysis, top, bottom)
            if save_dir is not None:
                fig.savefig(Path(save_dir) / f"{analysis.name} ({chart_name}).png", dpi=100)
            plt.show()
            plt.close(fig)                            # free the memory once it's been shown
    return all_change_ys


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find and graph lithology changes in mud-log images.")
    parser.add_argument("paths", nargs="+", help="a data folder (with images/ and bbox_annotation/), an images folder, or image files")
    parser.add_argument("--save", metavar="FOLDER", help="also save the graphs as PNGs in this folder")
    args = parser.parse_args()

    for path in map(Path, args.paths):
        if path.is_dir():
            plot_all(path, save_dir=args.save)
            continue

        result = analyse_image(path, annotation_dir=image_and_annotation_dirs(path.parent)[1])
        print(f"{result.name}: {result.scale.pixels_per_foot:.2f} px/ft from {result.scale.source}; "
              f"window {result.window_height}px, band {result.band_rows}px; {len(result.change_ys)} changes")
        print(interval_stats(result).round(2).to_string(index=False))
        figure = plot_analysis(result)
        if args.save:
            Path(args.save).mkdir(parents=True, exist_ok=True)
            figure.savefig(Path(args.save) / f"{result.name}.png", dpi=100)
        plt.show()
