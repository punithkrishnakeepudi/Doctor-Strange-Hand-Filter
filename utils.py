# small_funcs.py
import cv2 as cv
import numpy as np
from typing import List, Tuple

DEFAULT_LINE = (0, 140, 255)
WHITE = (255, 255, 255)

def get_positions(lmlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Return key points in a fixed order:
    wrist, thumb_tip, index_mcp, index_tip, middle_mcp, middle_tip, ring_tip, pinky_tip
    """
    if len(lmlist) < 21:
        raise ValueError("Landmarks need at least 21 points.")
    inds = [0, 4, 5, 8, 9, 12, 16, 20]
    return [lmlist[i] for i in inds]

def dist_euc(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    a = np.array(p1, dtype=np.float32)
    b = np.array(p2, dtype=np.float32)
    return float(np.linalg.norm(a - b))

def stroke_line(frame: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int],
                color: Tuple[int, int, int] = DEFAULT_LINE, thickness: int = 5) -> np.ndarray:
    # outer colored stroke
    cv.line(frame, p1, p2, color, thickness, lineType=cv.LINE_AA)
    # inner highlight
    inner_thick = max(1, thickness // 2)
    cv.line(frame, p1, p2, WHITE, inner_thick, lineType=cv.LINE_AA)
    return frame

def place_rgba(overlay: np.ndarray, frame: np.ndarray, x: int, y: int,
               size: Tuple[int, int] = None) -> np.ndarray:
    """
    Place RGBA overlay on BGR frame at position (x, y) (top-left).
    Resizes overlay to 'size' if provided.
    """
    if size:
        try:
            overlay = cv.resize(overlay, size, interpolation=cv.INTER_AREA)
        except cv.error as e:
            raise ValueError(f"Resize failed: {e}")

    if overlay.shape[-1] != 4:
        raise ValueError("Overlay must have 4 channels (RGBA).")

    h, w = overlay.shape[:2]
    H, W = frame.shape[:2]

    if x + w > W or y + h > H:
        # If overlay would go out of bounds, clip it instead of failing
        w = min(w, W - x)
        h = min(h, H - y)
        overlay = overlay[0:h, 0:w]

    b, g, r, a = cv.split(overlay)
    fg = cv.merge([b, g, r]).astype(np.float32)
    alpha = (a.astype(np.float32) / 255.0)[:, :, np.newaxis]

    roi = frame[y:y + h, x:x + w].astype(np.float32)
    out = alpha * fg + (1 - alpha) * roi
    frame[y:y + h, x:x + w] = out.astype(np.uint8)
    return frame
