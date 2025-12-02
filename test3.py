# main.py  (multi-color support; drop-in replacement)
import cv2 as cv
import mediapipe as mp
import json
import glob
import random
import os
from utils import get_positions, dist_euc, stroke_line, place_rgba
from typing import Tuple, List

CONFIG_PATH = "config.json"

def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def clamp(v: int, a: int, b: int) -> int:
    return max(a, min(v, b))

def open_camera(cfg: dict) -> cv.VideoCapture:
    cap = cv.VideoCapture(cfg["camera"]["device_id"])
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cfg["camera"]["width"])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cfg["camera"]["height"])
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera device.")
    return cap

def load_all_overlays(inner_folder: str, outer_folder: str) -> Tuple[List, List]:
    """
    Load all images from inner_folder and outer_folder.
    Returns two lists: inner_imgs, outer_imgs.
    """
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    def _gather(folder):
        files = []
        for p in patterns:
            files.extend(glob.glob(os.path.join(folder, p)))
        files = sorted(files)
        imgs = []
        for f in files:
            img = cv.imread(f, cv.IMREAD_UNCHANGED)
            if img is None:
                continue
            # ensure 4 channels
            if img.ndim == 2:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                b, g, r = cv.split(img)
                a = 255 * (b[:, :] == b[:, :])  # create alpha full
                img = cv.merge([b, g, r, a])
            imgs.append(img)
        return imgs

    inner_imgs = _gather(inner_folder)
    outer_imgs = _gather(outer_folder)
    return inner_imgs, outer_imgs

def process_single_hand(frame, hand_lms, cfg, inner_img, outer_img, rot_deg):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
    (wrist, thumb_tip, idx_mcp, idx_tip,
     mid_mcp, mid_tip, ring_tip, pinky_tip) = get_positions(pts)

    d_wrist_idx = dist_euc(wrist, idx_mcp)
    d_idx_pinky = dist_euc(idx_tip, pinky_tip)
    ratio = (d_idx_pinky / (d_wrist_idx + 1e-6))

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

    elif ratio >= 1.3:
        cx, cy = mid_mcp
        size = round(d_wrist_idx * cfg["overlay"]["shield_size_multiplier"])

        x1 = clamp(cx - size // 2, 0, w)
        y1 = clamp(cy - size // 2, 0, h)
        size = min(size, w - x1, h - y1)

        rot_deg = (rot_deg + cfg["overlay"]["rotation_degree_increment"]) % 360

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

    # optional explicit entry check
    expected = "main.py"
    if cfg.get("entry_point") != expected:
        raise RuntimeError(f"Config mismatch: entry_point expected '{expected}' but got '{cfg.get('entry_point')}'")

    cap = open_camera(cfg)

    # load all overlays from folders listed in config (instead of single file)
    inner_folder = os.path.dirname(cfg["overlay"]["inner_circle_path"]) or "Models/inner_circles"
    outer_folder = os.path.dirname(cfg["overlay"]["outer_circle_path"]) or "Models/outer_circles"

    inner_imgs, outer_imgs = load_all_overlays(inner_folder, outer_folder)
    if not inner_imgs or not outer_imgs:
        raise FileNotFoundError("No overlay images found in inner/outer folders. Put PNGs in Models/inner_circles and Models/outer_circles")

    hands = mp.solutions.hands.Hands(static_image_mode=False,
                                     max_num_hands=2,
                                     min_detection_confidence=0.6,
                                     min_tracking_confidence=0.5)

    # per-hand state: rotation degree and chosen image indexes
    rots = [0, 0]               # degrees for each hand
    chosen_idx = [random.randrange(len(inner_imgs)), random.randrange(len(outer_imgs))]  # pair index per-hand; we will use same pair for all hands initially

    print("Press 'n' to randomize ring colors, 'q' to quit.")

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
                # For each detected hand, ensure we have a chosen pair
                for i, hand_lms in enumerate(res.multi_hand_landmarks):
                    # ensure lists large enough
                    if i >= len(rots):
                        rots.append(0)
                    # pick per-hand images: choose by mapping i to indexes (or randomize)
                    # here we select random inner/outer per hand (you can change to cycle instead)
                    inner_img = inner_imgs[(chosen_idx[0] + i) % len(inner_imgs)]
                    outer_img = outer_imgs[(chosen_idx[1] + i) % len(outer_imgs)]

                    frame, rots[i] = process_single_hand(frame, hand_lms, cfg, inner_img, outer_img, rots[i])

                cv.putText(frame, f'Hands: {len(res.multi_hand_landmarks)}', (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv.imshow("Doctor Strange - MultiColor", frame)
            key = cv.waitKey(1) & 0xFF
            if key == ord(cfg["keybindings"]["quit_key"]):
                break
            if key == ord('n'):
                # randomize chosen base indexes so next frame shows new colors
                chosen_idx[0] = random.randrange(len(inner_imgs))
                chosen_idx[1] = random.randrange(len(outer_imgs))
                print("Randomized ring colors.")
    finally:
        cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main()
