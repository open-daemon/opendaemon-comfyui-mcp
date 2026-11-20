"""Where ComfyUI lives. One place, so the standalone release works on any PC.

Set COMFYUI_DIR (the folder that holds ComfyUI's main.py) and, if ComfyUI is not on the default address,
COMFYUI_URL. Without them we look in the usual places.
"""
import glob
import os
import urllib.parse

_HOME = os.path.expanduser("~")
_GUESSES = [
    r"C:\LLM\ComfyUI\ComfyUI_windows_portable\ComfyUI",
    os.path.join(_HOME, "ComfyUI_windows_portable", "ComfyUI"),
    os.path.join(_HOME, "ComfyUI"),
    os.path.join(_HOME, "Documents", "ComfyUI"),
    os.path.join(_HOME, "Desktop", "ComfyUI_windows_portable", "ComfyUI"),
] + sorted(glob.glob(r"C:\*\ComfyUI_windows_portable\ComfyUI")) + sorted(glob.glob(r"D:\*\ComfyUI_windows_portable\ComfyUI"))


def _find_comfy():
    d = os.environ.get("COMFYUI_DIR", "").strip().strip('"')
    if d:
        if os.path.isdir(os.path.join(d, "ComfyUI")) and not os.path.isfile(os.path.join(d, "main.py")):
            d = os.path.join(d, "ComfyUI")  # the portable folder was given instead of the ComfyUI folder inside it
        return os.path.abspath(d)
    for g in _GUESSES:
        if os.path.isfile(os.path.join(g, "main.py")):
            return g
    return _GUESSES[0]


COMFY = _find_comfy()
URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").strip().rstrip("/")
_u = urllib.parse.urlparse(URL)
HOST, PORT = _u.hostname or "127.0.0.1", _u.port or 8188
COMFY_ROOT = os.path.dirname(COMFY)  # ComfyUI_windows_portable for the portable build


def comfy_python():
    """The Python that runs ComfyUI: COMFYUI_PYTHON, else the portable build's own, else the venv next to it."""
    p = os.environ.get("COMFYUI_PYTHON", "").strip().strip('"')
    if p:
        return p
    for c in (os.path.join(COMFY_ROOT, "python_embeded", "python.exe"),
              os.path.join(COMFY, "venv", "Scripts", "python.exe"), os.path.join(COMFY, ".venv", "Scripts", "python.exe"),
              os.path.join(COMFY, "venv", "bin", "python"), os.path.join(COMFY, ".venv", "bin", "python")):
        if os.path.isfile(c):
            return c
    return None


HERE = os.path.dirname(os.path.abspath(__file__))


def _data_dir():
    """Where videos, workflows and settings are kept: COMFY_MCP_DATA, else next to the code when run from a
    folder (the dev copy), else a folder in the user's home when installed as a package."""
    d = os.environ.get("COMFY_MCP_DATA", "").strip().strip('"')
    if d:
        return os.path.abspath(os.path.expanduser(d))
    if "site-packages" in HERE.lower() or "dist-packages" in HERE.lower():
        return os.path.join(_HOME, "OpenDaemon-ComfyUI-MCP")
    return HERE


DATA = _data_dir()
OUT = os.path.join(DATA, "out")
os.makedirs(OUT, exist_ok=True)


def ffmpeg_candidates():
    """ffmpeg programs on this PC, best first: FFMPEG, CapCut's (Windows), the one on PATH, imageio-ffmpeg's
    (installed with this package), Remotion's (the dev PC)."""
    import shutil
    c = [os.environ.get("FFMPEG")] if os.environ.get("FFMPEG") else []
    c += sorted(glob.glob(os.path.join(os.environ.get("LOCALAPPDATA", ""), "CapCut", "Apps", "*", "ffmpeg.exe")), reverse=True)
    if shutil.which("ffmpeg"):
        c.append(shutil.which("ffmpeg"))
    try:
        import imageio_ffmpeg
        c.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    c.append(os.path.join(HERE, "..", "ad", "node_modules", "@remotion", "compositor-win32-x64-msvc", "ffmpeg.exe"))
    out = []
    for x in c:
        if x and os.path.exists(x) and x not in out:
            out.append(x)
    return out
