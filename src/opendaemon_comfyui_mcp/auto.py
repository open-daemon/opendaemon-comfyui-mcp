"""Auto mode in Python: the same settings the page's easySettings() builds (index.html).

Both read presets.json, so the page and the MCP server always agree. If you change
easySettings() in the page, change easy_settings() here too (test: tests/auto_parity).
"""
import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(HERE, "presets.json"), encoding="utf-8"))
PRESETS, BASE, STYLES, QUALITY, CAM = P["PRESETS"], P["BASE"], P["STYLES"], P["QUALITY"], P["CAM"]
AVOID = P["AVOID"]
SMALL = {**PRESETS["distilled"], "cfg": 1, "strength": 1, "cfps": 0, "fps": 24, "frames": 73,
         "max_shift": 2.05, "base_shift": 0.95, "sampler": "euler"}
# "full" stays as a kind for LTX dev checkpoints a user adds; the built-in 2B Dev was removed 11-18.
KIND_OF = {"ltxdist": "small", "ltxdev": "full", "ltx13": "ltx13", "ltx13dev": "ltx13dev", "vace": "vace"}
MODELS = {"small": "LTX-Video 2B Distilled - fast (about 6 min at Normal on a low-end GTX 1050 Ti 4 GB; a better card is faster)",
          "ltx13": "LTX-Video 13B Distilled (GGUF) - sharper, keeps the character, slower",
          "vace": "Wan 2.1 VACE 1.3B - 16 fps, keeps your picture as the first frame"}
SPEEDS = {"slow": 18, "normal": 24, "fast": 30}
LENGTHS = (1, 2, 3, 4, 5, 6, 7)  # seconds the page offers
CAM_RE = re.compile(r"\s*(Static camera|Slow camera (zoom in|zoom out|pan to the right|pan to the left))\.?", re.I)


def _round(x):  # JavaScript Math.round
    return int(math.floor(x + 0.5))


def _index(items, value, what):
    """A style/quality by number or by (part of) its name."""
    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        i = int(value)
        if 0 <= i < len(items):
            return i
    else:
        v = str(value).lower().strip()
        for i, it in enumerate(items):
            if it["name"].lower() == v:
                return i
        for i, it in enumerate(items):
            if v and v in it["name"].lower():
                return i
    raise ValueError("Unknown %s %r. Choose one of: %s" % (what, value, "; ".join(it["name"] for it in items)))


def style_defaults(style=0):
    st = STYLES[_index(STYLES, style, "style")]
    s = {**SMALL, **st["set"]}
    return {"movement": int(s["comp"]), "seconds": (int(s["frames"]) - 1) // 24, "fps": int(s["fps"])}


def easy_settings(image="", prompt="", model="small", custom=None, style=0, quality=1, movement=None,
                  seconds=None, speed=None, camera="still", seed=-1, avoid=None, width=768, height=416):
    """model: small | full | ltx13 | vace (a built-in), or the kind of an added model when custom is set.
    custom: an entry from custom-models.json ({name, file, type, kind}) or None.
    image: a picture in ComfyUI's input folder, or "" to make the video from the prompt alone."""
    si, qi = _index(STYLES, style, "style"), _index(QUALITY, quality, "quality")
    st, q = STYLES[si], QUALITY[qi]
    d = style_defaults(si)
    movement = d["movement"] if movement is None else int(movement)
    frames = d["seconds"] * 24 + 1 if seconds is None else int(seconds) * 24 + 1
    fps = d["fps"] if speed is None else (SPEEDS[speed] if speed in SPEEDS else int(speed))
    if camera not in CAM:
        raise ValueError("Unknown camera %r. Choose one of: %s" % (camera, ", ".join(CAM)))
    avoid = AVOID if avoid is None else avoid

    s = {**BASE, **SMALL, **st["set"], "image": image or "", "prompt": prompt, "seed": seed, "negative": ""}
    s["prompt"] = CAM_RE.sub("", s["prompt"]).strip() + " " + CAM[camera]
    s["comp"], s["frames"], s["fps"], s["cfps"], s["seed"] = movement, frames, fps, 24, int(seed)
    if float(st["set"].get("cfps", 0)) > 24:
        s["cfps"] = st["set"]["cfps"]
    if st.get("add") and st["add"] not in s["prompt"]:
        s["prompt"] = s["prompt"].strip() + " " + st["add"]
    s["width"], s["height"] = (int(width), int(height)) if q.get("manual") else (q["w"], q["h"])
    s["refine"] = bool(q.get("refine"))
    m, cm, secs = (KIND_OF[custom["type"]] if custom else model), custom, (frames - 1) / 24
    if m not in ("small", "full", "ltx13", "ltx13dev", "vace"):
        raise ValueError("Unknown model %r. Choose one of: %s" % (model, ", ".join(MODELS)))
    if m in ("full", "ltx13dev"):
        s.update(PRESETS["dev"], cfg=3, steps=30, schedule="auto", negative=avoid, comp=s["comp"])
        s["cfps"] = 24  # the full model looked "underwater" when told a higher fps
    if cm and m in ("small", "full"):
        s["checkpoint"] = cm["file"]
    if m in ("ltx13", "ltx13dev"):  # model-only LTX: the VAE comes from the 2B distilled checkpoint
        s.update(family="ltx13", unet=cm["file"] if cm else "LTXV-13B-0.9.8-distilled-Q3_K_M.gguf",
                 checkpoint="ltxv-2b-0.9.8-distilled-fp8.safetensors")
    if m == "vace":
        sizes = {0: (512, 288), 1: (640, 352), 2: (832, 480), 3: (640, 352)}
        w, h = (s["width"], s["height"]) if q.get("manual") else sizes[qi]
        s.update(family="vace", unet=cm["file"] if cm else "wan2.1_vace_1.3B_fp16.safetensors",
                 width=w, height=h, steps=20, cfg=6, sampler="uni_pc", scheduler="simple", shift=8, refine=False,
                 negative=avoid, checkpoint=cm["name"] if cm else "Wan 2.1 VACE 1.3B")
        s["frames"] = _round(secs * 16) + 1
        s["fps"] = _round(s["fps"] * 16 / 24)
    s["easy"] = ("★ " + cm["name"] + " · ") if cm else {"full": "Full · ", "ltx13": "LTX 13B · ", "vace": "Wan VACE · "}.get(m, "")
    s["easy"] += st["name"] + " · " + q["name"].split(" ")[0]
    return s


def guess_minutes(s):
    """Rough render time on a 4 GB card (GTX 1050 Ti class), from our own runs. Bigger cards are faster."""
    px = int(s["width"]) * int(s["height"]) * int(s["frames"]) / (768 * 416 * 73)
    base = {"vace": 8, "ltx13": 14}.get(s.get("family"), 6)
    if int(s.get("steps", 8)) >= 30:
        base *= 4
    t = base * px
    if s.get("refine"):
        t = t * 1.2 + 25
    return max(1, _round(t))
