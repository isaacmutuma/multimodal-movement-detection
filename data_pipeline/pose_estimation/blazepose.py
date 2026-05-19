"""
MediaPipe BlazePose landmark extraction (Tasks API).

Uses ``PoseLandmarker`` with the ``pose_landmarker_lite`` model bundle — not the
legacy ``mp.solutions.pose`` API. Returns 33 body landmarks per detected person
as plain data structures (no drawing; see ``skeleton_overlay`` for visuals).
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Optional, Sequence, Union

import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision
from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmark as MpPoseLandmark

NUM_POSE_LANDMARKS = 33

POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

DEFAULT_MODEL_FILENAME = "pose_landmarker_lite.task"


class PoseLandmarkIndex(IntEnum):
    """
    BlazePose landmark indices (0–32).

    Matches MediaPipe's ``PoseLandmark`` enum so you can use names like
    ``PoseLandmarkIndex.RIGHT_WRIST`` instead of remembering index ``16``.
    """

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Convenience alias requested in project spec (index 16).
RIGHT_WRIST = PoseLandmarkIndex.RIGHT_WRIST

LandmarkSelector = Union[int, PoseLandmarkIndex, MpPoseLandmark, str]


@dataclass(frozen=True)
class PoseLandmarkPoint:
    """
    One body landmark in normalized image coordinates.

    Attributes:
        x: Horizontal position in [0, 1] relative to image width.
        y: Vertical position in [0, 1] relative to image height.
        z: Depth relative to the hip midpoint (smaller = closer to camera).
        visibility: MediaPipe visibility score in [0, 1]; higher = more
            confident the landmark is visible in the frame.
    """

    x: float
    y: float
    z: float
    visibility: float


def _validate_frame(frame: np.ndarray, name: str = "frame") -> None:
    """
    Ensure ``frame`` is a non-empty 2D grayscale or 3-channel image array.

    Raises:
        TypeError: If ``frame`` is not a ``numpy.ndarray``.
        ValueError: If shape or channel count is invalid.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(frame).__name__}")
    if frame.size == 0:
        raise ValueError(f"{name} is empty")
    if frame.ndim == 2:
        return
    if frame.ndim == 3 and frame.shape[2] == 3:
        return
    raise ValueError(
        f"{name} must be (H, W) or (H, W, 3), got shape {frame.shape}"
    )


def default_model_path(weights_dir: Optional[Path] = None) -> Path:
    """
    Return the default on-disk path for the lite pose landmarker model.

    Args:
        weights_dir: Directory for ``.task`` files; defaults to project
            ``weights/`` next to the repository root.

    Returns:
        Path ending in ``pose_landmarker_lite.task``.
    """
    if weights_dir is None:
        weights_dir = Path(__file__).resolve().parents[2] / "weights"
    return weights_dir / DEFAULT_MODEL_FILENAME


def download_pose_landmarker_lite(
    dest_path: Optional[Path] = None,
    *,
    force: bool = False,
) -> Path:
    """
    Download Google's ``pose_landmarker_lite.task`` bundle if missing.

    The Tasks API does not auto-download models; this function fetches the
    official float16 lite bundle (~few MB) once and reuses it on later runs.

    Args:
        dest_path: Where to save the file; uses ``default_model_path()`` if
            ``None``.
        force: Re-download even when the file already exists.

    Returns:
        Resolved path to the downloaded ``.task`` file.

    Raises:
        urllib.error.URLError: Network or HTTP failure.
        OSError: Filesystem errors while writing.
    """
    path = Path(dest_path) if dest_path is not None else default_model_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.is_file() and not force:
        return path.resolve()

    print(f"Downloading pose_landmarker_lite to {path} ...")
    try:
        urllib.request.urlretrieve(POSE_LANDMARKER_LITE_URL, path)
    except urllib.error.URLError as exc:
        raise urllib.error.URLError(
            f"Failed to download model from {POSE_LANDMARKER_LITE_URL}: {exc}"
        ) from exc

    return path.resolve()


def frame_to_rgb_uint8(
    frame: np.ndarray,
    *,
    color_format: str = "bgr",
) -> np.ndarray:
    """
    Convert a frame to RGB ``uint8`` for MediaPipe ``Image(SRGB)``.

    OpenCV webcams and ``VideoCapture`` use BGR by default. MediaPipe expects
    SRGB when you pass ``image_format=mp.ImageFormat.SRGB``.

    Args:
        frame: Grayscale ``(H, W)`` or color ``(H, W, 3)``.
        color_format: ``"bgr"`` (default) or ``"rgb"`` — how to interpret
            three-channel input.

    Returns:
        RGB array of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        ValueError: If ``color_format`` is not ``bgr`` or ``rgb``.
    """
    _validate_frame(frame)
    color_format = color_format.lower()
    if color_format not in ("bgr", "rgb"):
        raise ValueError(f"color_format must be 'bgr' or 'rgb', got {color_format!r}")

    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    elif color_format == "bgr":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = frame.copy()

    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(rgb)


def numpy_frame_to_mp_image(
    frame: np.ndarray,
    *,
    color_format: str = "bgr",
) -> mp.Image:
    """
    Wrap a NumPy frame as a MediaPipe ``Image`` in SRGB format.

    Args:
        frame: Input image array.
        color_format: ``"bgr"`` or ``"rgb"`` for three-channel frames.

    Returns:
        ``mediapipe.Image`` ready for ``PoseLandmarker.detect``.
    """
    rgb = frame_to_rgb_uint8(frame, color_format=color_format)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _normalized_landmark_to_point(lm) -> PoseLandmarkPoint:
    """
    Map a MediaPipe ``NormalizedLandmark`` to ``PoseLandmarkPoint``.

    Uses ``visibility`` when present; otherwise falls back to ``presence``.
    """
    visibility = lm.visibility
    if visibility is None:
        visibility = lm.presence if lm.presence is not None else 0.0
    return PoseLandmarkPoint(
        x=float(lm.x),
        y=float(lm.y),
        z=float(lm.z),
        visibility=float(visibility),
    )


def _result_to_landmarks(
    result: vision.PoseLandmarkerResult,
) -> Optional[List[PoseLandmarkPoint]]:
    """
    Parse ``PoseLandmarkerResult`` into 33 landmarks for the first person.

    Returns:
        List of 33 ``PoseLandmarkPoint`` values, or ``None`` if no pose detected.
    """
    if not result.pose_landmarks:
        return None

    first_pose = result.pose_landmarks[0]
    if len(first_pose) != NUM_POSE_LANDMARKS:
        raise RuntimeError(
            f"Expected {NUM_POSE_LANDMARKS} landmarks, got {len(first_pose)}"
        )

    return [_normalized_landmark_to_point(lm) for lm in first_pose]


def _resolve_landmark_index(selector: LandmarkSelector) -> int:
    """
    Convert an index, enum member, or landmark name string to an integer index.

    Raises:
        TypeError: Unsupported selector type.
        ValueError: Unknown name or index out of range.
    """
    if isinstance(selector, int):
        index = selector
    elif isinstance(selector, (PoseLandmarkIndex, MpPoseLandmark)):
        index = int(selector)
    elif isinstance(selector, str):
        key = selector.strip().upper()
        try:
            index = int(PoseLandmarkIndex[key])
        except KeyError as exc:
            raise ValueError(
                f"Unknown landmark name {selector!r}. "
                f"Use e.g. 'RIGHT_WRIST' or an index 0–{NUM_POSE_LANDMARKS - 1}."
            ) from exc
    else:
        raise TypeError(
            f"selector must be int, PoseLandmarkIndex, PoseLandmark name, or str; "
            f"got {type(selector).__name__}"
        )

    if not 0 <= index < NUM_POSE_LANDMARKS:
        raise ValueError(
            f"Landmark index must be in [0, {NUM_POSE_LANDMARKS - 1}], got {index}"
        )
    return index


def get_landmark(
    landmarks: Sequence[PoseLandmarkPoint],
    selector: LandmarkSelector,
) -> PoseLandmarkPoint:
    """
    Return one landmark from a full 33-point list by index or name.

    Examples:
        ``get_landmark(landmarks, 16)``
        ``get_landmark(landmarks, PoseLandmarkIndex.RIGHT_WRIST)``
        ``get_landmark(landmarks, "RIGHT_WRIST")``

    Args:
        landmarks: Sequence of length 33 from ``detect_pose_landmarks``.
        selector: Integer index, ``PoseLandmarkIndex``, MediaPipe ``PoseLandmark``,
            or case-insensitive name string.

    Returns:
        The selected ``PoseLandmarkPoint``.

    Raises:
        ValueError: Wrong list length or invalid selector.
        TypeError: Invalid selector type.
    """
    if len(landmarks) != NUM_POSE_LANDMARKS:
        raise ValueError(
            f"landmarks must contain {NUM_POSE_LANDMARKS} points, got {len(landmarks)}"
        )
    index = _resolve_landmark_index(selector)
    return landmarks[index]


class PoseLandmarkerSession:
    """
    Stateful wrapper around MediaPipe ``PoseLandmarker`` (IMAGE mode).

    Loads the ``.task`` model once and runs ``detect`` on each frame. Use as a
    context manager so native resources are released when done.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        *,
        color_format: str = "bgr",
        num_poses: int = 1,
        min_pose_detection_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        auto_download: bool = True,
    ) -> None:
        """
        Configure the landmarker; model file is resolved on ``open()``.

        Args:
            model_path: Path to ``pose_landmarker_lite.task``; downloaded to
                ``weights/`` when missing and ``auto_download`` is True.
            color_format: Default ``"bgr"`` or ``"rgb"`` for ``detect`` inputs.
            num_poses: Maximum number of poses (pipeline uses the first).
            min_pose_detection_confidence: Detection score threshold.
            min_pose_presence_confidence: Landmark presence threshold.
            min_tracking_confidence: Tracking threshold (IMAGE mode).
            auto_download: Download lite model if ``model_path`` is missing.
        """
        self._model_path = (
            Path(model_path) if model_path is not None else default_model_path()
        )
        self._color_format = color_format
        self._num_poses = num_poses
        self._min_pose_detection_confidence = min_pose_detection_confidence
        self._min_pose_presence_confidence = min_pose_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._auto_download = auto_download
        self._landmarker: Optional[vision.PoseLandmarker] = None

    def open(self) -> "PoseLandmarkerSession":
        """
        Create the underlying ``PoseLandmarker`` from the ``.task`` file.

        Returns:
            ``self`` for chaining.

        Raises:
            FileNotFoundError: Model missing and ``auto_download`` is False.
        """
        if not self._model_path.is_file():
            if self._auto_download:
                download_pose_landmarker_lite(self._model_path)
            else:
                raise FileNotFoundError(
                    f"Pose landmarker model not found: {self._model_path}"
                )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=self._num_poses,
            min_pose_detection_confidence=self._min_pose_detection_confidence,
            min_pose_presence_confidence=self._min_pose_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self

    def close(self) -> None:
        """Release the MediaPipe landmarker."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> "PoseLandmarkerSession":
        return self.open()

    def __exit__(self, *args: object) -> None:
        self.close()

    def detect(
        self,
        frame: np.ndarray,
        *,
        color_format: Optional[str] = None,
    ) -> Optional[List[PoseLandmarkPoint]]:
        """
        Detect 33 pose landmarks in a single frame.

        Args:
            frame: BGR or RGB ``(H, W, 3)`` image, or grayscale ``(H, W)``.
            color_format: Overrides session default for this call.

        Returns:
            List of 33 ``PoseLandmarkPoint`` for the first detected person, or
            ``None`` if no person is found.

        Raises:
            RuntimeError: Session not opened with ``open()`` or context manager.
        """
        if self._landmarker is None:
            raise RuntimeError(
                "PoseLandmarkerSession is not open. Use 'with PoseLandmarkerSession() "
                "as session:' or call open() first."
            )

        _validate_frame(frame)
        fmt = color_format if color_format is not None else self._color_format
        mp_image = numpy_frame_to_mp_image(frame, color_format=fmt)
        result = self._landmarker.detect(mp_image)
        return _result_to_landmarks(result)


def detect_pose_landmarks(
    frame: np.ndarray,
    *,
    model_path: Optional[Path] = None,
    color_format: str = "bgr",
    session: Optional[PoseLandmarkerSession] = None,
) -> Optional[List[PoseLandmarkPoint]]:
    """
    One-shot pose detection on a single frame.

    Convenience wrapper: opens a landmarker, runs ``detect``, and closes it.
    For video, create one ``PoseLandmarkerSession`` and reuse it per frame.

    Args:
        frame: Input image as NumPy array.
        model_path: Optional path to the ``.task`` model file.
        color_format: ``"bgr"`` or ``"rgb"``.
        session: Existing open session; if provided, ``model_path`` is ignored
            and the session is not closed after this call.

    Returns:
        33 landmarks or ``None`` if no person detected.
    """
    if session is not None:
        return session.detect(frame, color_format=color_format)

    with PoseLandmarkerSession(model_path=model_path, color_format=color_format) as s:
        return s.detect(frame, color_format=color_format)


def _synthetic_blank_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """
    Build a blank BGR frame with no person (for negative detection tests).

    Returns:
        Black ``uint8`` image of shape ``(height, width, 3)``.
    """
    return np.zeros((height, width, 3), dtype=np.uint8)


def _synthetic_landmark_list() -> List[PoseLandmarkPoint]:
    """
    Build a fake 33-point list to test ``get_landmark`` without a real image.

    Returns:
        Landmarks with distinct ``x`` per index for easy assertions.
    """
    return [
        PoseLandmarkPoint(
            x=i / NUM_POSE_LANDMARKS,
            y=0.5,
            z=0.0,
            visibility=1.0,
        )
        for i in range(NUM_POSE_LANDMARKS)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test BlazePose landmark extraction (MediaPipe Tasks API)."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional image path with a visible person (positive detection test).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download pose_landmarker_lite.task even if present.",
    )
    args = parser.parse_args()

    print("=== BlazePose module tests ===\n")

    # --- Unit-style tests (no model required) ---
    print("1. Frame validation and color conversion")
    blank = _synthetic_blank_frame()
    _validate_frame(blank)
    rgb = frame_to_rgb_uint8(blank, color_format="bgr")
    assert rgb.shape == blank.shape and rgb.dtype == np.uint8
    print("   OK — blank synthetic frame, RGB conversion")

    print("\n2. get_landmark by index and name")
    fake = _synthetic_landmark_list()
    wrist_by_idx = get_landmark(fake, 16)
    wrist_by_name = get_landmark(fake, "RIGHT_WRIST")
    wrist_by_enum = get_landmark(fake, PoseLandmarkIndex.RIGHT_WRIST)
    assert wrist_by_idx.x == wrist_by_name.x == wrist_by_enum.x
    assert abs(wrist_by_idx.x - 16 / NUM_POSE_LANDMARKS) < 1e-6
    print(f"   OK — RIGHT_WRIST x={wrist_by_idx.x:.4f}, visibility={wrist_by_idx.visibility}")

    # --- Integration: model download + inference ---
    print("\n3. Model download and PoseLandmarker (lite)")
    model_path = download_pose_landmarker_lite(force=args.force_download)
    print(f"   Model at: {model_path} ({model_path.stat().st_size // 1024} KiB)")

    print("\n4. Detection on synthetic blank frame (expect no person)")
    with PoseLandmarkerSession(model_path=model_path, auto_download=False) as session:
        landmarks_blank = session.detect(blank)
    assert landmarks_blank is None
    print("   OK — no landmarks on empty frame (as expected)")

    if args.image is not None:
        print(f"\n5. Detection on image: {args.image}")
        if not args.image.is_file():
            raise FileNotFoundError(
                f"Image not found: {args.image}\n"
                "Use a real file on your Mac, e.g. --image ~/Pictures/photo.jpg "
                "(not the example placeholder /path/to/photo.jpg)."
            )
        image_bgr = cv2.imread(str(args.image))
        if image_bgr is None:
            raise ValueError(f"Could not read image: {args.image}")
        with PoseLandmarkerSession(model_path=model_path, auto_download=False) as session:
            landmarks_image = session.detect(image_bgr)
        if landmarks_image is None:
            print("   No person detected.")
        else:
            nose = get_landmark(landmarks_image, "NOSE")
            wrist = get_landmark(landmarks_image, RIGHT_WRIST)
            print(f"   OK — 33 landmarks; NOSE=({nose.x:.3f}, {nose.y:.3f}), "
                  f"RIGHT_WRIST=({wrist.x:.3f}, {wrist.y:.3f})")
    else:
        print(
            "\n5. Skipped image test — pass --image /path/to/photo.jpg "
            "to verify detection on a real person."
        )

    print("\nAll built-in tests passed.")
