# Multimodal Movement Detection

A cross-modal transformer that takes video and audio as joint inputs and learns
their relationship during training — not two separate models fused at inference.

Built at IDOLL Robotics (Busan) as part of an internship research project.

---

## What it does

Two target applications share the same architecture:

**Tic detection (clinical).** A low-resource tool for detecting tic disorders
from webcam and microphone. Optical flow and pose estimation capture motor tics;
MFCC features capture vocal tics. The model learns that a wrist movement and a
throat-clearing within ~200 ms are jointly more significant than either alone.
Target settings: clinical environments where specialist equipment is scarce.

**Human-robot interaction (industrial).** The same architecture on IDOLL's
robot platform: lip movement, expression, pose, and gesture fused with speech
features for intent understanding that is more robust than audio-only models.

---

## Architecture

```
INPUT
├── Video frames (T × H × W × 3)
│   └── Visual Encoder: MobileNetV2 (pretrained, fine-tuned)
│       └── Output: (batch, T, D=256)
│
└── Audio frames (T × 26 MFCC coefficients)
    └── Audio Encoder: 1D CNN (trained from scratch)
        └── Output: (batch, T, D=256)

FUSION
└── Cross-Modal Transformer (2–4 stacked blocks)
    ├── Video tokens attend to audio tokens  (nn.MultiheadAttention)
    ├── Audio tokens attend to video tokens  (nn.MultiheadAttention)
    └── Output: unified audio-visual representation (batch, T, D)

TASK HEAD
├── Clinical:   TicDetectionHead → binary tic / no-tic
└── HRI:        HRIHead → speech intent / gesture / emotion state
```

Inspired by MulT (Tsai et al., ACL 2019) and AV-HuBERT (Meta AI, 2022).

---

## Current status — weeks 1–5 of 11 (data pipeline phase)

The model, training loop, and inference runtime are not yet written. What exists
is the data pipeline that will feed the model.

### Implemented

| Module | File | What it does |
|--------|------|-------------|
| Farneback optical flow | `data_pipeline/optical_flow/farneback.py` | Dense per-pixel motion vectors between consecutive frames; magnitude map; HSV colour visualisation; magnitude-heatmap overlay; webcam live demo |
| BlazePose landmark extraction | `data_pipeline/pose_estimation/blazepose.py` | 33 body landmarks via MediaPipe Tasks API (`PoseLandmarkerSession` context manager); auto-downloads the lite model; `get_landmark` by index or name |

### Scaffolded stubs (not yet implemented)

- `data_pipeline/optical_flow/visualize.py`
- `data_pipeline/pose_estimation/trajectory.py`, `skeleton_overlay.py`
- `data_pipeline/roi_flow/roi_optical_flow.py`
- `data_pipeline/audio/capture.py`, `mfcc_extraction.py`
- `model/` — visual encoder, audio encoder, cross-modal transformer, task heads
- `training/` — train loop, evaluation, dataset
- `detection/` — threshold, detector, logger
- `hri/` — lip tracking, HRI inference

---

## Setup

```bash
pip install -r requirements.txt
```

The BlazePose model (`pose_landmarker_lite.task`) is auto-downloaded to
`weights/` on first run — it does not need to be committed to the repo.

---

## Running what exists

**Farneback — synthetic self-test** (no camera required):
```bash
python data_pipeline/optical_flow/farneback.py
```
Runs flow on two synthetic frames with a shifted white square, prints shape and
magnitude stats, opens two visualisation windows. Press any key to close.

**Farneback — live webcam demo**:
```bash
python data_pipeline/optical_flow/farneback.py --webcam
# optional: --camera 1  (if default index 0 is wrong)
```
Displays HSV flow and magnitude-overlay windows. Press `q`/Esc in a window or
Ctrl-C in the terminal to stop.

**BlazePose — unit and integration tests**:
```bash
python data_pipeline/pose_estimation/blazepose.py
```
Validates frame conversion, `get_landmark` (by index, name, and enum), downloads
the lite model if absent, and runs detection on a synthetic blank frame.

**BlazePose — with a real image**:
```bash
python data_pipeline/pose_estimation/blazepose.py --image /path/to/photo.jpg
```
Prints normalized `(x, y)` coordinates for NOSE and RIGHT_WRIST.

---

## Roadmap

| Weeks | Focus |
|-------|-------|
| 1–5 | Data pipeline — optical flow, pose, ROI flow, threshold detection |
| 6 | Audio — microphone capture and MFCC extraction |
| 7 | Audio encoder (1D CNN), trained standalone |
| 8 | Cross-modal transformer and unified training |
| 9 | HRI — lip tracking, HRI head, IDOLL presentation |
| 10–11 | Integration, demo video, technical write-up |

Target completion: July 31, 2026.

---

## References

- Tsai et al., "Multimodal Transformer for Unaligned Multimodal Language Sequences," ACL 2019
- Shi et al., "Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction," ICLR 2022
- Conelea et al., "Automated Tic Detection from Video," Movement Disorders, 2024
