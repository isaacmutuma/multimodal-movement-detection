"""
Farneback dense optical flow between consecutive video frames.

Computes per-pixel motion vectors with OpenCV's Farneback algorithm, then
derives a magnitude map (how much each pixel moved). Optional HSV colouring
maps motion direction to hue and speed to brightness.
"""

from __future__ import annotations

import argparse
from typing import Optional, Tuple

import cv2
import numpy as np


def _validate_frame(frame: np.ndarray, name: str) -> None:
    """
    Check that a frame is a non-empty 2D or 3D NumPy array.

    Raises:
        TypeError: If ``frame`` is not a ``numpy.ndarray``.
        ValueError: If the array has no elements or an unsupported rank.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame).__name__}")
    if frame.size == 0:
        raise ValueError(f"{name} is empty")
    if frame.ndim not in (2, 3):
        raise ValueError(
            f"{name} must be 2D (grayscale) or 3D (color), got shape {frame.shape}"
        )


def _validate_frame_pair(prev_frame: np.ndarray, next_frame: np.ndarray) -> None:
    """
    Ensure two frames share the same spatial shape before optical flow.

    Raises:
        ValueError: If height/width (and channel count) do not match.
    """
    _validate_frame(prev_frame, "prev_frame")
    _validate_frame(next_frame, "next_frame")
    if prev_frame.shape[:2] != next_frame.shape[:2]:
        raise ValueError(
            "prev_frame and next_frame must have the same height and width: "
            f"{prev_frame.shape[:2]} vs {next_frame.shape[:2]}"
        )


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert a color or grayscale frame to single-channel uint8.

    OpenCV expects grayscale for ``calcOpticalFlowFarneback``. Color frames
    in BGR order (OpenCV default from ``VideoCapture``) are converted with
    ``cv2.COLOR_BGR2GRAY``. Already-grayscale inputs are returned unchanged
    when dtype is uint8; other dtypes are cast safely.

    Args:
        frame: Grayscale ``(H, W)`` or color ``(H, W, 3)`` array (BGR).

    Returns:
        Grayscale array of shape ``(H, W)`` and dtype ``uint8``.

    Raises:
        TypeError, ValueError: See ``_validate_frame``.
    """
    _validate_frame(frame, "frame")

    if frame.ndim == 2:
        gray = frame
    else:
        if frame.shape[2] != 3:
            raise ValueError(
                f"Color frame must have 3 channels, got shape {frame.shape}"
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray


def compute_farneback(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    *,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 15,
    iterations: int = 3,
    poly_n: int = 5,
    poly_sigma: float = 1.2,
    flags: int = 0,
) -> np.ndarray:
    """
    Run Farneback dense optical flow between two consecutive frames.

    Farneback builds a Gaussian pyramid of both frames, estimates flow at
    coarse scales, then refines at finer scales. Each pixel gets a 2D motion
    vector ``(dx, dy)`` in image coordinates.

    Args:
        prev_frame: Earlier frame (grayscale or color).
        next_frame: Later frame (same size as ``prev_frame``).
        pyr_scale: Image scale between pyramid levels (< 1).
        levels: Number of pyramid levels.
        winsize: Averaging window size per pixel.
        iterations: Iterations of the polynomial expansion at each level.
        poly_n: Size of the pixel neighbourhood for polynomial expansion.
        poly_sigma: Gaussian standard deviation for ``poly_n``.
        flags: OpenCV Farneback flags (e.g. use initial flow).

    Returns:
        Flow field of shape ``(H, W, 2)`` with ``flow[..., 0]`` = dx and
        ``flow[..., 1]`` = dy (``float32``).

    Raises:
        TypeError, ValueError: Invalid frames or mismatched sizes.
    """
    _validate_frame_pair(prev_frame, next_frame)

    prev_gray = to_grayscale(prev_frame)
    next_gray = to_grayscale(next_frame)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
    return flow


def flow_to_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Convert a flow field to per-pixel motion magnitude.

    Magnitude is the Euclidean length of each ``(dx, dy)`` vector:
    ``sqrt(dx**2 + dy**2)``. Brighter values in downstream visualisations
    mean faster motion at that pixel.

    Args:
        flow: Array of shape ``(H, W, 2)``.

    Returns:
        Magnitude array of shape ``(H, W)``, dtype ``float32``.

    Raises:
        ValueError: If ``flow`` does not have shape ``(H, W, 2)``.
    """
    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(
            f"flow must have shape (H, W, 2), got {getattr(flow, 'shape', None)}"
        )

    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    return magnitude


def compute_flow_magnitude(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    **farneback_kwargs,
) -> np.ndarray:
    """
    Compute optical flow magnitude between two frames (main pipeline API).

    This is the function other modules should call: pass two consecutive
    frames, receive an ``(H, W)`` map of how much each pixel moved.

    Args:
        prev_frame: Earlier frame.
        next_frame: Later frame.
        **farneback_kwargs: Forwarded to ``compute_farneback``.

    Returns:
        Magnitude array of shape ``(H, W)``, dtype ``float32``.
    """
    flow = compute_farneback(prev_frame, next_frame, **farneback_kwargs)
    return flow_to_magnitude(flow)


def visualize_flow_hsv(
    flow: np.ndarray,
    magnitude_clip: Optional[float] = None,
) -> np.ndarray:
    """
    Colour-code flow direction and speed using the HSV wheel.

    Hue encodes motion direction (angle of the flow vector). Value (brightness)
    encodes speed (magnitude), clipped for stable display. Saturation is fixed
    at maximum so colours stay vivid.

    Args:
        flow: Flow field of shape ``(H, W, 2)``.
        magnitude_clip: Upper bound for normalising magnitude; if ``None``,
            uses the 99th percentile of magnitudes in this frame.

    Returns:
        BGR image of shape ``(H, W, 3)``, dtype ``uint8``, suitable for
        ``cv2.imshow`` or video writers.
    """
    magnitude = flow_to_magnitude(flow)
    angle = np.arctan2(flow[..., 1], flow[..., 0])

    if magnitude_clip is None:
        magnitude_clip = float(np.percentile(magnitude, 99)) or 1.0
    magnitude_clip = max(magnitude_clip, 1e-6)

    magnitude_norm = np.clip(magnitude / magnitude_clip, 0.0, 1.0)

    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (magnitude_norm * 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def overlay_magnitude_on_frame(
    frame: np.ndarray,
    magnitude: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend a magnitude heatmap on top of the original frame for debugging.

    Args:
        frame: BGR or grayscale frame used as the background.
        magnitude: ``(H, W)`` magnitude map from ``compute_flow_magnitude``.
        alpha: Blend weight for the heatmap (0 = only frame, 1 = only heatmap).

    Returns:
        BGR image with the heatmap overlaid.
    """
    if frame.ndim == 2:
        background = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        background = frame.copy()

    mag_norm = magnitude / (magnitude.max() + 1e-6)
    heatmap = cv2.applyColorMap(
        (mag_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    return cv2.addWeighted(background, 1.0 - alpha, heatmap, alpha, 0.0)


def _synthetic_frame_pair(
    height: int = 240, width: int = 320, shift: int = 8
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two BGR frames with a bright square shifted horizontally (for tests).

    Returns:
        ``(prev_frame, next_frame)`` as uint8 BGR arrays.
    """
    prev = np.zeros((height, width, 3), dtype=np.uint8)
    next_frame = np.zeros((height, width, 3), dtype=np.uint8)
    y0, y1 = height // 4, 3 * height // 4
    x0, x1 = width // 4, width // 4 + 60
    prev[y0:y1, x0:x1] = (255, 255, 255)
    next_frame[y0:y1, x0 + shift : x1 + shift] = (255, 255, 255)
    return prev, next_frame


def _run_webcam_demo(camera_index: int = 0) -> None:
    """Read consecutive webcam frames and display flow magnitude + HSV."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError("Could not read an initial frame from the webcam")

    print("Webcam demo — press 'q' to quit.")
    try:
        while True:
            ret, next_frame = cap.read()
            if not ret:
                break

            flow = compute_farneback(prev_frame, next_frame)
            magnitude = flow_to_magnitude(flow)
            hsv_vis = visualize_flow_hsv(flow)
            overlay = overlay_magnitude_on_frame(next_frame, magnitude)

            cv2.imshow("Farneback HSV", hsv_vis)
            cv2.imshow("Magnitude overlay", overlay)

            prev_frame = next_frame
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Farneback optical flow on synthetic or webcam frames."
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Run live demo on default webcam instead of synthetic frames.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Webcam device index (used with --webcam).",
    )
    args = parser.parse_args()

    if args.webcam:
        _run_webcam_demo(args.camera)
    else:
        prev, nxt = _synthetic_frame_pair()
        flow = compute_farneback(prev, nxt)
        magnitude = compute_flow_magnitude(prev, nxt)

        print("Synthetic frame test")
        print(f"  prev shape:      {prev.shape}")
        print(f"  next shape:      {nxt.shape}")
        print(f"  flow shape:      {flow.shape}")
        print(f"  magnitude shape: {magnitude.shape}")
        print(f"  magnitude max:   {magnitude.max():.4f}")
        print(f"  magnitude mean:  {magnitude.mean():.4f}")

        assert magnitude.shape == prev.shape[:2]
        assert magnitude.max() > 0.0, "Expected non-zero motion on shifted square"

        hsv_vis = visualize_flow_hsv(flow)
        overlay = overlay_magnitude_on_frame(nxt, magnitude)

        cv2.imshow("Synthetic — HSV flow", hsv_vis)
        cv2.imshow("Synthetic — magnitude overlay", overlay)
        print("Press any key in an image window to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print("All synthetic checks passed.")
