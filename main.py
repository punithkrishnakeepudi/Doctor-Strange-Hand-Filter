# main.py
import cv2 as cv
import mediapipe as mp
import json
from utils import get_positions, dist_euc, stroke_line, place_rgba
from typing import Tuple

CONFIG_PATH = "config.json"

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def clamp(v: int, a: int, b: int) -> int:
    """Clamp integer between a and b inclusive."""
    return max(a, min(v, b))

def open_camera(cfg: dict) -> cv.VideoCapture:
    cap = cv.VideoCapture(cfg["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device.")
    return cap

def load_overlay_images(cfg: dict) -> Tuple:
    inner = cv.imread(cfg["overlay"]["inner_circle_path"], cv.IMREAD_UNCHANGED)
    outer = cv.imread(cfg["overlay"]["outer_circle_path"], cv.IMREAD_UNCHANGED)
    if inner is None or outer is None:
        raise FileNotFoundError("One or more overlay images not found.")
    return inner, outer

def process_frame(frame, hands, cfg, inner_img, outer_img, rot_deg):
    h, w = frame.shape[:2]
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if res.multi_hand_landmarks:
        for hand_lms in res.multi_hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
            (wrist, thumb_tip, idx_mcp, idx_tip,
             mid_mcp, mid_tip, ring_tip, pinky_tip) = get_positions(pts)

            d_wrist_idx = dist_euc(wrist, idx_mcp)
            d_idx_pinky = dist_euc(idx_tip, pinky_tip)
            # ratio of spread: bigger = more open
            ratio = (d_idx_pinky / (d_wrist_idx + 1e-6))

            # quick visual debug: draw small hub at palm center
            if ratio <= 0.0:
                ratio = 0.0

            # when hand is slightly open: draw finger-to-wrist lines
            if 0.5 < ratio < 1.3:
                fingers = [thumb_tip, idx_tip, mid_tip, ring_tip, pinky_tip]
                for f in fingers:
                    frame = stroke_line(frame, wrist, f,
                                        color=tuple(cfg["line_settings"]["color"]),
                                        thickness=cfg["line_settings"]["thickness"])
                for i in range(len(fingers) - 1):
                    frame = stroke_line(frame, fingers[i], fingers[i + 1],
                                        color=tuple(cfg["line_settings"]["color"]),
                                        thickness=cfg["line_settings"]["thickness"])

            # when open wide enough: overlay shield (dual rings)
            elif ratio >= 1.3:
                cx, cy = mid_mcp
                size = round(d_wrist_idx * cfg["overlay"]["shield_size_multiplier"])

                # clamp top-left so overlay fits inside frame
                x1 = clamp(cx - size // 2, 0, w)
                y1 = clamp(cy - size // 2, 0, h)
                size = min(size, w - x1, h - y1)

                # update rotation
                rot_deg = (rot_deg + cfg["overlay"]["rotation_degree_increment"]) % 360

                # rotate overlays (inner opposite direction)
                M_outer = cv.getRotationMatrix2D((outer_img.shape[1] // 2, outer_img.shape[0] // 2),
                                                rot_deg, 1.0)
                M_inner = cv.getRotationMatrix2D((inner_img.shape[1] // 2, inner_img.shape[0] // 2),
                                                -rot_deg, 1.0)

                rotated_outer = cv.warpAffine(outer_img, M_outer,
                                             (outer_img.shape[1], outer_img.shape[0]),
                                             flags=cv.INTER_LINEAR,
                                             borderMode=cv.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))
                rotated_inner = cv.warpAffine(inner_img, M_inner,
                                             (inner_img.shape[1], inner_img.shape[0]),
                                             flags=cv.INTER_LINEAR,
                                             borderMode=cv.BORDER_CONSTANT,
                                             borderValue=(0, 0, 0, 0))

                frame = place_rgba(rotated_outer, frame, x1, y1, (size, size))
                frame = place_rgba(rotated_inner, frame, x1, y1, (size, size))

    return frame, rot_deg

def main():
    cfg = load_config()

    # verify entry point explicitly
    expected = "main.py"
    if cfg.get("entry_point") != expected:
        raise RuntimeError(f"Config mismatch: entry_point expected '{expected}' but got '{cfg.get('entry_point')}'")

    cap = open_camera(cfg)
    inner_img, outer_img = load_overlay_images(cfg)

    hands = mp.solutions.hands.Hands(static_image_mode=False,
                                    max_num_hands=1,
                                    min_detection_confidence=0.6,
                                    min_tracking_confidence=0.5)

    deg = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed, exiting.")
                break

            frame = cv.flip(frame, 1)
            frame, deg = process_frame(frame, hands, cfg, inner_img, outer_img, deg)

            cv.imshow("Doctor Strange Variant", frame)
            if cv.waitKey(1) == ord(cfg["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
