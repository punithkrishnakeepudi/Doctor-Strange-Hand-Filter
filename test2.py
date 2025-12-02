# main.py  (multi-hand enabled)
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

def process_single_hand(frame, hand_lms, cfg, inner_img, outer_img, rot_deg):
    """
    Process one detected hand (landmarks), apply effects and return new frame and updated rot_deg.
    rot_deg is the rotation value for this specific hand (degrees).
    """
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
    (wrist, thumb_tip, idx_mcp, idx_tip,
     mid_mcp, mid_tip, ring_tip, pinky_tip) = get_positions(pts)

    d_wrist_idx = dist_euc(wrist, idx_mcp)
    d_idx_pinky = dist_euc(idx_tip, pinky_tip)
    ratio = (d_idx_pinky / (d_wrist_idx + 1e-6))

    # when hand slightly open: draw lines
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

        # update this hand's rotation
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

        # place both rings centered at calculated top-left (x1,y1) with size x size
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

    # enable up to 2 hands
    hands = mp.solutions.hands.Hands(static_image_mode=False,
                                     max_num_hands=2,
                                     min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)

    # maintain a rotation degree for each possible hand (index 0 and 1)
    rots = [0, 0]

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                print("Frame capture failed, exiting.")
                break

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                # iterate over all detected hands and process each one
                for i, hand_lms in enumerate(res.multi_hand_landmarks):
                    # ensure rots list long enough (should be size 2)
                    if i >= len(rots):
                        rots.append(0)
                    frame, rots[i] = process_single_hand(frame, hand_lms, cfg, inner_img, outer_img, rots[i])

                # optional: show count
                cv.putText(frame, f'Hands: {len(res.multi_hand_landmarks)}', (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("Doctor Strange Variant - Multihand", frame)
            if cv.waitKey(1) == ord(cfg["keybindings"]["quit_key"]):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
