"""Live YOLO webcam showcase with a neon-style dashboard.

Requires:
- opencv-python
- ultralytics

The model file yolo26n.pt is downloaded automatically by Ultralytics if it is
not already available locally.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
from ultralytics import YOLO


CAMERA_WINDOW_NAME = "YOLO Live Showcase - Camera"
CHART_WINDOW_NAME = "YOLO Live Showcase - Barplot"
CAMERA_INDEX = 0
MODEL_PATH = "yolo26n.pt"
DEFAULT_CONFIDENCE = 0.35
DEFAULT_FRAME_SKIP = 1
PANEL_WIDTH = 520
PANEL_HEIGHT = 720
CAMERA_WINDOW_WIDTH = 1280
CAMERA_WINDOW_HEIGHT = 720
CHART_WINDOW_WIDTH = 820
CHART_WINDOW_HEIGHT = 980


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live webcam object detection with a dramatic dashboard."
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Webcam index")
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="YOLO model path or name",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=DEFAULT_CONFIDENCE,
        help="Minimum confidence for displayed detections",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=DEFAULT_FRAME_SKIP,
        help="Run YOLO once every N frames",
    )
    return parser.parse_args()


def build_background(width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        blend = y / max(height - 1, 1)
        color = np.array([22, 16, 38]) * (1 - blend) + np.array([8, 8, 14]) * blend
        canvas[y, :] = color.astype(np.uint8)

    for offset in range(-height, width, 42):
        start = (max(offset, 0), max(-offset, 0))
        end = (min(width, offset + height), min(height, height + offset))
        cv2.line(canvas, start, end, (35, 24, 62), 1, cv2.LINE_AA)

    overlay = canvas.copy()
    cv2.circle(overlay, (width - 90, 110), 120, (90, 40, 180), -1, cv2.LINE_AA)
    cv2.circle(overlay, (70, height - 90), 150, (35, 160, 220), -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.20, canvas, 0.80, 0)


def make_canvas(width: int = PANEL_WIDTH, height: int = PANEL_HEIGHT) -> np.ndarray:
    canvas = build_background(width, height)
    cv2.rectangle(canvas, (18, 18), (width - 18, height - 18), (120, 80, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (24, 24), (width - 24, height - 24), (18, 18, 28), 1, cv2.LINE_AA)
    return canvas


def draw_text(canvas: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def fit_to_box(image: np.ndarray, box_width: int, box_height: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((box_height, box_width, 3), dtype=np.uint8)

    height, width = image.shape[:2]
    scale = min(box_width / max(width, 1), box_height / max(height, 1))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    bordered = np.zeros((box_height, box_width, 3), dtype=np.uint8)
    x_offset = (box_width - new_width) // 2
    y_offset = (box_height - new_height) // 2
    bordered[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return bordered


def color_for_index(index: int) -> tuple[int, int, int]:
    palette = [
        (80, 220, 255),
        (255, 120, 90),
        (120, 255, 140),
        (240, 180, 60),
        (200, 110, 255),
        (90, 180, 255),
        (255, 90, 190),
    ]
    return palette[index % len(palette)]


def summarize_detections(result, confidence_threshold: float) -> Counter:
    counts: Counter = Counter()
    if result.boxes is None or len(result.boxes) == 0:
        return counts

    class_ids = result.boxes.cls.int().tolist()
    confidences = result.boxes.conf.tolist()
    for class_id, confidence in zip(class_ids, confidences):
        if confidence >= confidence_threshold:
            class_name = result.names[int(class_id)]
            counts[class_name] += 1
    return counts


def draw_bar_panel(canvas: np.ndarray, counts: Counter, elapsed_seconds: float, frame_index: int) -> None:
    panel_w = canvas.shape[1]
    draw_text(canvas, "OBJECT RADAR", (36, 62), 1.0, (255, 255, 255), 2)
    draw_text(canvas, f"session {elapsed_seconds:05.1f}s", (36, 96), 0.55, (170, 190, 255), 1)
    draw_text(canvas, f"frame {frame_index}", (36, 122), 0.55, (170, 190, 255), 1)

    if not counts:
        draw_text(canvas, "No confident detections yet.", (36, 180), 0.7, (220, 220, 220), 1)
        draw_text(canvas, "Move in front of the camera.", (36, 212), 0.6, (140, 200, 255), 1)
        return

    sorted_items = counts.most_common(10)
    max_value = max(value for _, value in sorted_items)
    chart_left = 36
    chart_top = 170
    chart_right = panel_w - 34
    chart_width = chart_right - chart_left
    row_height = 44
    bar_max_width = chart_width - 150

    for row_index, (name, value) in enumerate(sorted_items):
        top = chart_top + row_index * row_height
        bottom = top + 24
        bar_width = int(bar_max_width * (value / max_value))
        color = color_for_index(row_index)

        cv2.rectangle(canvas, (chart_left, top), (chart_left + bar_max_width, bottom), (34, 34, 48), -1)
        cv2.rectangle(canvas, (chart_left, top), (chart_left + bar_width, bottom), color, -1)
        cv2.rectangle(canvas, (chart_left, top), (chart_left + bar_max_width, bottom), (70, 70, 90), 1)

        draw_text(canvas, name[:18], (chart_left + 6, top - 6), 0.55, (245, 245, 245), 1)
        draw_text(canvas, str(value), (chart_right - 78, top + 18), 0.6, (255, 255, 255), 1)

    total_seen = sum(counts.values())
    unique_objects = len(counts)
    draw_text(canvas, f"unique objects: {unique_objects}", (36, canvas.shape[0] - 78), 0.6, (180, 255, 180), 1)
    draw_text(canvas, f"total detections: {total_seen}", (36, canvas.shape[0] - 48), 0.6, (180, 255, 180), 1)


def draw_footer(canvas: np.ndarray, counts: Counter) -> None:
    footer_y = canvas.shape[0] - 18
    if counts:
        objects_text = "  |  ".join(f"{name}:{value}" for name, value in counts.most_common(6))
    else:
        objects_text = "waiting for the first detection"
    draw_text(canvas, objects_text[:78], (36, footer_y), 0.5, (255, 210, 120), 1)


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    camera = cv2.VideoCapture(args.camera)

    if not camera.isOpened():
        raise RuntimeError(f"Nao consegui abrir a camara no indice {args.camera}.")

    cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CHART_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CAMERA_WINDOW_NAME, CAMERA_WINDOW_WIDTH, CAMERA_WINDOW_HEIGHT)
    cv2.resizeWindow(CHART_WINDOW_NAME, CHART_WINDOW_WIDTH, CHART_WINDOW_HEIGHT)

    frame_index = 0
    last_counts: Counter = Counter()
    last_annotated = None
    start_time = datetime.now()

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        display_frame = cv2.resize(frame, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)

        if frame_index % max(args.frame_skip, 1) == 0:
            results = model.predict(display_frame, conf=args.confidence, verbose=False)
            result = results[0]
            last_counts = summarize_detections(result, args.confidence)
            last_annotated = result.plot()
        elif last_annotated is None:
            last_annotated = display_frame.copy()

        annotated = last_annotated.copy() if last_annotated is not None else display_frame.copy()
        camera_canvas = fit_to_box(annotated, CAMERA_WINDOW_WIDTH, CAMERA_WINDOW_HEIGHT)

        chart_canvas = make_canvas(CHART_WINDOW_WIDTH, CHART_WINDOW_HEIGHT)
        draw_text(chart_canvas, "LIVE BARPLOT", (36, 34), 0.82, (255, 255, 255), 2)
        draw_text(chart_canvas, "object frequency scoreboard", (36, 150), 0.68, (160, 200, 255), 1)
        draw_bar_panel(chart_canvas, last_counts, (datetime.now() - start_time).total_seconds(), frame_index)

        cv2.imshow(CAMERA_WINDOW_NAME, camera_canvas)
        cv2.imshow(CHART_WINDOW_NAME, chart_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        frame_index += 1

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()