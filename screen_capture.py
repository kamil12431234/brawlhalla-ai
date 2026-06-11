import os
import logging
import subprocess
import tempfile
import numpy as np
import cv2
from config import CAPTURE_WIDTH, CAPTURE_HEIGHT, REGION_LEFT, REGION_TOP

logger = logging.getLogger(__name__)

_sct = None
_use_wayland: bool | None = None


def _detect_backend() -> str:
    global _use_wayland

    if _use_wayland is not None:
        return "wayland" if _use_wayland else "x11"

    session = os.environ.get("XDG_SESSION_TYPE", "").lower()
    if session == "wayland":
        _use_wayland = True
        return "wayland"
    if os.environ.get("WAYLAND_DISPLAY"):
        _use_wayland = True
        return "wayland"

    _use_wayland = False
    return "x11"


def _capture_x11(monitor: dict) -> np.ndarray:
    global _sct
    import mss
    if _sct is None:
        _sct = mss.mss()
    shot = _sct.grab(monitor)
    return cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)


def _capture_wayland(region: dict) -> np.ndarray:
    left = region["left"]
    top = region["top"]
    width = region["width"]
    height = region["height"]
    geom = f"{left},{top} {width}x{height}"

    fd, tmpfile = tempfile.mkstemp(suffix=".ppm", prefix="bh_", dir="/dev/shm")
    os.close(fd)

    try:
        subprocess.run(
            ["grim", "-g", geom, "-t", "ppm", tmpfile],
            check=True,
            timeout=2,
            capture_output=True,
        )
        img = cv2.imread(tmpfile, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"grim produced unreadable image: {tmpfile}")
        return img
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"grim capture failed: {e.stderr.decode().strip()}") from e
    finally:
        try:
            os.unlink(tmpfile)
        except OSError:
            pass


def capture_frame(full_screen: bool = False) -> np.ndarray:
    region = {
        "left": REGION_LEFT,
        "top": REGION_TOP,
        "width": CAPTURE_WIDTH,
        "height": CAPTURE_HEIGHT,
    }

    use_wayland = _use_wayland
    if use_wayland is None:
        backend = _detect_backend()
        use_wayland = (backend == "wayland")

    if use_wayland:
        return _capture_wayland(region)
    else:
        return _capture_x11(region)
