#!/usr/bin/env python3
"""
Utilities for MediaPipe face detection.
Provides a wrapper that returns pixel-space bounding boxes from BGR frames.
"""
from __future__ import annotations
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional

try:
    import mediapipe as mp
except Exception as e:  # pragma: no cover
    raise SystemExit("[Error] mediapipe is not installed. Install with: pip install mediapipe==0.10.14") from e


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    score: float

    @property
    def xyxy(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


class MPFaceDetector:
    """MediaPipe Face Detection wrapper.

    - Input: BGR frame (numpy array)
    - Output: List[Detection] in pixel coordinates.
    """
    def __init__(self, model_selection: int = 0, min_confidence: float = 0.5) -> None:
        # model_selection: 0 -> short-range (2m), 1 -> full-range (5m)
        self.model_selection = int(model_selection)
        self.min_confidence = float(min_confidence)
        self._mp_fd = mp.solutions.face_detection
        self._detector = self._mp_fd.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )

    def detect(self, frame_bgr) -> List[Detection]:
        if frame_bgr is None:
            return []
        h, w = frame_bgr.shape[:2]
        # MediaPipe expects RGB input
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._detector.process(rgb)
        dets: List[Detection] = []
        if not res or not res.detections:
            return dets
        for d in res.detections:
            # Relative bbox
            rel = d.location_data.relative_bounding_box
            x = max(0, int(rel.xmin * w))
            y = max(0, int(rel.ymin * h))
            bw = max(0, int(rel.width * w))
            bh = max(0, int(rel.height * h))
            score = 0.0
            if d.score:
                try:
                    score = float(d.score[0])
                except Exception:
                    pass
            # Clip to frame
            x = min(x, w - 1)
            y = min(y, h - 1)
            if x + bw > w:
                bw = w - x
            if y + bh > h:
                bh = h - y
            if bw <= 0 or bh <= 0:
                continue
            dets.append(Detection(x=x, y=y, w=bw, h=bh, score=score))
        return dets

    def close(self) -> None:
        try:
            self._detector.close()
        except Exception:
            pass


# ---------------- Face Mesh Utilities ----------------
class MPFaceMesh:
    """MediaPipe Face Mesh wrapper with simple draw helper.

    Usage:
      fm = MPFaceMesh(max_faces=1, refine_landmarks=True)
      res = fm.process(frame_bgr)
      fm.draw(frame_bgr, res)
    """
    def __init__(
        self,
        max_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_fm = mp.solutions.face_mesh
        self._mp_draw = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._mesh = self._mp_fm.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr):
        if frame_bgr is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self._mesh.process(rgb)

    def draw(self, frame_bgr, results) -> None:
        if results is None or not getattr(results, "multi_face_landmarks", None):
            return
        for face_landmarks in results.multi_face_landmarks:
            self._mp_draw.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self._mp_fm.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_tesselation_style(),
            )
            self._mp_draw.draw_landmarks(
                image=frame_bgr,
                landmark_list=face_landmarks,
                connections=self._mp_fm.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self._mp_styles.get_default_face_mesh_contours_style(),
            )

    def close(self) -> None:
        try:
            self._mesh.close()
        except Exception:
            pass
