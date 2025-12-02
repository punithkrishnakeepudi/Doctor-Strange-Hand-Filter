# Doctor Strange Hand Filter 🔮

A lightweight computer-vision project that recreates the iconic Doctor Strange magical shield effect in real time using MediaPipe Hand Tracking and OpenCV.

The repository includes single-hand and dual-hand demos, customizable overlays, and modular utility functions.

---

📥 Clone & Installation Guide

1️⃣ Clone the repository

```bash
git clone https://github.com/punithkrishnakeepudi/Doctor-Strange-Hand-Filter.git
```

Replace `<your-username>/<your-repo-name>` with your actual GitHub repository path.

Move into the project folder:

```bash
cd Doctor-Strange-Hand-Filter
```

2️⃣ Create and activate a virtual environment

Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If MediaPipe fails to install, check the official MediaPipe installation docs or install a wheel for your Python version.

---

🚀 Project Overview

This repo provides scripts demonstrating different variations of the effect:

| Script   | Purpose                                           |
|----------|---------------------------------------------------|
| `main.py`  | Core single-hand Doctor Strange effect           |
| `test2.py`  | Dual-hand version (both hands tracked)         |
| `test3.py`  | Dual-hand demo with one hand using a blue variant overlay |
| `utils.py`  | Helper functions for overlays, drawing, and distance calculations |
| `Models/`  | All inner/outer circular PNG overlay assets      |
| `config.json`  | Optional configuration file for adjusting behavior |

---

📁 Project Structure

    Doctor_Strange_hand_filter/
    │
    ├── Models/
    │   ├── inner_circles/
    │   │   ├── blue.png
    │   │   ├── light_orange.png
    │   │   └── orange.png
    │   ├── outer_circles/
    │   │   ├── dark_red.png
    │   │   ├── orange.png
    │   │   └── red.png
    │
    ├── config.json
    ├── main.py
    ├── test2.py
    ├── test3.py
    ├── utils.py
    ├── requirements.txt
    └── README.md

Important: Keep the folder names and structure the same. The scripts load images using these exact paths.

---

🧩 Requirements

Python 3.8+ (3.10 verified)

Packages in `requirements.txt`:

- opencv-python
- mediapipe
- numpy

---

▶️ Running the demos

Make sure your webcam is plugged in and free (no other apps using it).

Single-hand effect:

```bash
python main.py
```

Dual-hand effect:

```bash
python test2.py
```

Dual-hand effect with blue secondary ring:

```bash
python test3.py
```

If your PC uses a different webcam index, change:

```python
cv2.VideoCapture(0)
```

to:

```python
cv2.VideoCapture(1)
```

---

⚙️ Configuration (optional)

`config.json` allows customization:

- Circular overlay image paths
- Rotation speed
- Magic shield size multiplier
- Hand-line highlight colors
- Quit key binding

If `config.json` is missing, scripts fall back to defaults.

---

🧠 Script Details

`main.py`

- Tracks one hand
- Computes distances between key landmarks
- Detects "hand open" gesture
- Draws connecting lines
- Overlays rotating inner + outer circles

`test2.py`

- Expands the system to two hands
- Each hand gets its own overlay logic

`test3.py`

- Same as `test2.py`
- Applies a blue effect on one hand for visual contrast

`utils.py`

Provides:

- Landmark extraction
- Euclidean distance calculations
- Line-drawing helpers
- RGBA overlay compositing

---

🎨 Asset Customization

You can replace any PNG inside:

- `Models/inner_circles/`
- `Models/outer_circles/`

Tips:

- Use transparent (RGBA) PNGs
- Higher resolution → cleaner effect
- Add multiple colors for more variation

---

🛠 Troubleshooting

No webcam feed?

- Check camera permissions
- Close other apps (Zoom, OBS, Discord)
- Try changing camera index

Mediapipe errors?

```bash
pip install --upgrade pip
pip install mediapipe
```

Low FPS?

Reduce resolution:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
```

Turn off debug drawings to improve runtime performance.

---

🔧 Tips & Expansions

You can easily extend the project by:

- Adding particle effects
- Auto-cycling ring colors
- Recording video output
- Building a GUI to switch themes

If you want any of these features, tell me — I can generate the code.

---

📄 License

This repository does not include a license. Add one if you intend to distribute or publish this work.

---

🤝 Contributing

Feel free to open issues or PRs if you want to add:

- New magic effects
- Alternative color themes
- Higher-quality ring assets
- Performance improvements

---

✨ Enjoy the magic!

If you want a banner image, GIF preview, or badges for the top of the README, just say “add visuals” and I’ll generate them.
