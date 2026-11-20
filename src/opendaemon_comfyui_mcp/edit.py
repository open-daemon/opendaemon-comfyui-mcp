"""Editing finished videos with ffmpeg (phase 3 of the MCP). No GPU needed, except where noted in server.py.

Every edit writes a NEW file in out/ (the original is never changed) with a .json next to it that keeps the
source's render settings plus {"edit": ...}, so the page's gallery shows edits like any other video.
Text (titles, captions, cards) is drawn as pictures with Pillow and laid over the video, so any ffmpeg build works
(many have no drawtext/subtitles filter).
"""
import glob
import json
import os
import shutil
import subprocess
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
import paths
OUT = paths.OUT
TMP = os.path.join(OUT, "tmp")
VOICE = os.path.join(OUT, "voice")
SHAPES = {"9:16": (1080, 1920), "1:1": (1080, 1080), "16:9": (1920, 1080), "4:5": (1080, 1350)}
PLATFORMS = {  # shape, max MB, max seconds (our own defaults, kept under each platform's upload limit)
    "tiktok": ("9:16", 280, 600), "reels": ("9:16", 250, 90), "shorts": ("9:16", 250, 180),
    "facebook": ("16:9", 10, 240),  # our 10 MB Facebook copy
    "youtube": ("16:9", 2000, 3600), "x": ("16:9", 500, 140), "square": ("1:1", 100, 600)}


class EditError(Exception):
    pass


# ---------- ffmpeg ----------
def _candidates():
    return paths.ffmpeg_candidates()


_FF = {}


def ff():
    """The ffmpeg with the most of the filters we use (xfade, overlay, minterpolate...)."""
    if not _FF:
        need = ("xfade", "overlay", "minterpolate", "sidechaincompress", "gblur", "zoompan", "colortemperature", "lutyuv")
        best, score = None, -1
        for f in _candidates():
            try:
                fl = subprocess.run([f, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=20).stdout
                en = subprocess.run([f, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
            except Exception:
                continue
            s = sum((" %s " % n) in fl for n in need)
            if s > score:
                best, score = f, s
                _FF.update(path=f, filters=fl, encoders=[e for e in ("libx264", "h264_nvenc", "h264_mf") if e in en])
        if not best:
            raise EditError("No ffmpeg found. Install ffmpeg (ffmpeg.org) or set FFMPEG to its path.")
        probe = os.path.join(os.path.dirname(best), "ffprobe.exe" if os.name == "nt" else "ffprobe")
        _FF["probe"] = probe if os.path.exists(probe) else shutil.which("ffprobe")
    return _FF


def _run(args, what):
    r = subprocess.run([ff()["path"], "-hide_banner", "-v", "error", "-y"] + args, capture_output=True, text=True, timeout=1800)
    if r.returncode:
        raise EditError("%s failed: %s" % (what, (r.stderr or "")[-700:]))


def _venc(bitrate):
    """Video encoder args, best first; _encode tries the next one if a hardware encoder can't start."""
    out = []
    for e in ff()["encoders"]:
        if e == "libx264":
            out.append(["-c:v", "libx264", "-preset", "medium", "-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.5)), "-bufsize", str(bitrate * 2)])
        elif e == "h264_nvenc":
            out.append(["-c:v", "h264_nvenc", "-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.5)), "-bufsize", str(bitrate * 2)])
        else:
            out.append(["-c:v", e, "-b:v", str(bitrate)])
    return out


def _encode(args_in, filt, dst, bitrate, audio=True, extra=()):
    err = None
    for enc in _venc(bitrate):
        cmd = list(args_in) + (["-filter_complex", filt] if filt else []) + list(extra) + enc + ["-pix_fmt", "yuv420p"]
        cmd += (["-c:a", "aac", "-b:a", "160k"] if audio else ["-an"]) + ["-movflags", "+faststart", dst]
        try:
            _run(cmd, "Encoding")
            return dst
        except EditError as e:
            err = e
    raise err or EditError("No H.264 encoder in this ffmpeg.")


def probe(path):
    p = ff()["probe"]
    if not p:
        raise EditError("ffprobe not found next to ffmpeg.")
    r = subprocess.run([p, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path], capture_output=True, text=True)
    if r.returncode:
        raise EditError("Can't read %s: %s" % (os.path.basename(path), r.stderr[-300:]))
    d = json.loads(r.stdout)
    v = next((s for s in d["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in d["streams"] if s["codec_type"] == "audio"), None)
    out = {"duration": float(d["format"].get("duration") or 0), "audio": a is not None}
    if v:
        num, den = (v.get("avg_frame_rate") or v.get("r_frame_rate") or "24/1").split("/")
        out.update(width=int(v["width"]), height=int(v["height"]), fps=round(float(num) / float(den or 1), 3) if float(num) else 24.0)
    return out


def _bitrate(w, h, fps):
    return int(max(4e6, min(20e6, w * h * fps * 0.22)))


# ---------- files ----------
def resolve(v):
    """A video name in out/ or a full path to any file."""
    if not v:
        raise EditError("No video given.")
    if os.path.isabs(v) and os.path.isfile(v):
        return v
    p = os.path.join(OUT, os.path.basename(v))
    if os.path.isfile(p):
        return p
    raise EditError("No video called %r (use a name from list_videos, or a full path)." % v)


def resolve_any(f, what="file"):
    if f and os.path.isfile(f):
        return os.path.abspath(f)
    for d in (OUT, VOICE, os.path.join(paths.DATA, "music"), os.path.join(HERE, "music")):
        p = os.path.join(d, os.path.basename(f or ""))
        if f and os.path.isfile(p):
            return p
    raise EditError("No %s at %r. Give a full path." % (what, f))


import threading
TRACK = threading.local()  # build_video collects the in-between files it makes, to move them out of the gallery


def _new(op):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "ed_%s_%s_%s.mp4" % (time.strftime("%m%d_%H%M%S"), op, uuid.uuid4().hex[:4]))
    if getattr(TRACK, "files", None) is not None:
        TRACK.files.append(p)
    return p


def _finish(dst, op, sources, params, extra_meta=None):
    """Writes the .json next to an edit: the first source's settings + what was done."""
    base = {}
    js = os.path.splitext(sources[0])[0] + ".json" if sources else None
    if js and os.path.exists(js):
        try:
            base = json.load(open(js))
        except Exception:
            base = {}
    for k in ("render_seconds", "review", "gpu"):
        base.pop(k, None)
    prev = base.get("easy") or ""
    meta = dict(base, edit={"op": op, "from": [os.path.basename(s) for s in sources], "params": params},
                easy="Edit: %s%s" % (op, (" of " + prev) if prev and not prev.startswith("Edit:") else ""))
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(dst[:-4] + ".json", "w"), indent=1)
    info = probe(dst)
    return {"video": os.path.basename(dst), "path": dst, "seconds": round(info["duration"], 2),
            "width": info.get("width"), "height": info.get("height"), "mb": round(os.path.getsize(dst) / 1e6, 2)}


def _meta(src):
    js = os.path.splitext(src)[0] + ".json"
    try:
        return json.load(open(js)) if os.path.exists(js) else {}
    except Exception:
        return {}


def _aud_or_silence(src, info, idx_label, dur=None):
    """An audio input spec: the file's own audio, or silence of the same length (so joins always have audio)."""
    if info["audio"]:
        return [], "[%d:a]" % idx_label
    return ["-f", "lavfi", "-t", "%.3f" % (dur or info["duration"]), "-i", "anullsrc=r=44100:cl=stereo"], None


# ---------- simple edits ----------
def trim(v, start=0.0, end=None):
    src = resolve(v)
    info = probe(src)
    end = info["duration"] if end is None or end <= 0 else min(float(end), info["duration"])
    start = max(0.0, float(start))
    if end - start < 0.2:
        raise EditError("The part to keep is shorter than 0.2 s (the video is %.2f s)." % info["duration"])
    dst = _new("trim")
    _encode(["-ss", "%.3f" % start, "-to", "%.3f" % end, "-i", src], None, dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"])
    return _finish(dst, "trim", [src], {"start": start, "end": end})


def _atempo(x):
    parts = []
    while x > 2.0:
        parts.append("atempo=2.0")
        x /= 2.0
    while x < 0.5:
        parts.append("atempo=0.5")
        x /= 0.5
    parts.append("atempo=%.4f" % x)
    return ",".join(parts)


def speed(v, x=1.0):
    x = float(x)
    if not 0.1 <= x <= 8:
        raise EditError("Speed must be between 0.1 (10x slower) and 8 (8x faster).")
    src = resolve(v)
    info = probe(src)
    f = "[0:v]setpts=PTS/%.5f[v]" % x + (";[0:a]%s[a]" % _atempo(x) if info["audio"] else "")
    dst = _new("speed")
    _encode(["-i", src], f, dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "[a]"] if info["audio"] else []))
    return _finish(dst, "speed", [src], {"x": x})


def reverse(v):
    src = resolve(v)
    info = probe(src)
    f = "[0:v]reverse[v]" + (";[0:a]areverse[a]" if info["audio"] else "")
    dst = _new("reverse")
    _encode(["-i", src], f, dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "[a]"] if info["audio"] else []))
    return _finish(dst, "reverse", [src], {})


def loop(v, times=2):
    times = max(2, min(int(times), 20))
    src = resolve(v)
    info = probe(src)
    dst = _new("loop")
    _encode(["-stream_loop", str(times - 1), "-i", src], None, dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"])
    return _finish(dst, "loop", [src], {"times": times})


def boomerang(v, times=1):
    src = resolve(v)
    info = probe(src)
    f = "[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[o]"
    if times > 1:
        f += ";[o]loop=loop=%d:size=%d:start=0[v]" % (int(times) - 1, int(info["duration"] * info["fps"] * 2) + 2)
    else:
        f = f.replace("[o]", "[v]")
    dst = _new("boomerang")
    _encode(["-i", src], f, dst, _bitrate(info["width"], info["height"], info["fps"]), False, ["-map", "[v]"])
    return _finish(dst, "boomerang", [src], {"times": times})


def fade(v, fade_in=0.5, fade_out=0.5):
    src = resolve(v)
    info = probe(src)
    d = info["duration"]
    vf, af = [], []
    if fade_in > 0:
        vf.append("fade=t=in:st=0:d=%.3f" % fade_in)
        af.append("afade=t=in:st=0:d=%.3f" % fade_in)
    if fade_out > 0:
        vf.append("fade=t=out:st=%.3f:d=%.3f" % (max(0, d - fade_out), fade_out))
        af.append("afade=t=out:st=%.3f:d=%.3f" % (max(0, d - fade_out), fade_out))
    f = "[0:v]%s[v]" % (",".join(vf) or "null") + (";[0:a]%s[a]" % (",".join(af) or "anull") if info["audio"] else "")
    dst = _new("fade")
    _encode(["-i", src], f, dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "[a]"] if info["audio"] else []))
    return _finish(dst, "fade", [src], {"in": fade_in, "out": fade_out})


def color_adjust(v, brightness=0.0, contrast=1.0, saturation=1.0, warmth=0.0):
    """brightness -1..1, contrast 0.5..2, saturation 0..3, warmth -1 (cool) .. 1 (warm)."""
    src = resolve(v)
    info = probe(src)
    b, c, s = float(brightness) * 64, float(contrast), float(saturation)
    chain = ["lutyuv=y='clip((val-128)*%.3f+128+%.1f,0,255)':u='clip((val-128)*%.3f+128,0,255)':v='clip((val-128)*%.3f+128,0,255)'" % (c, b, s, s)]
    if warmth:
        chain.append("colortemperature=temperature=%d" % int(6500 - float(warmth) * 2500))
    dst = _new("color")
    _encode(["-i", src], "[0:v]%s[v]" % ",".join(chain), dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    return _finish(dst, "color", [src], {"brightness": brightness, "contrast": contrast, "saturation": saturation, "warmth": warmth})


def crop_zoom(v, zoom=1.25, direction="in", x=0.5, y=0.5):
    """A slow camera move on a finished clip. direction: in | out | left | right | up | down | none (= a still crop).
    x, y (0..1) = the point to zoom on."""
    src = resolve(v)
    info = probe(src)
    w, h, fps, n = info["width"], info["height"], info["fps"], max(1, int(info["duration"] * info["fps"]))
    z = max(1.01, min(float(zoom), 4.0))
    zx = {"in": "1+(%f-1)*on/%d" % (z, n), "out": "%f-(%f-1)*on/%d" % (z, z, n)}.get(direction, "%f" % z)
    cx, cy = "(iw-iw/zoom)*%f" % x, "(ih-ih/zoom)*%f" % y
    if direction == "left":
        cx = "(iw-iw/zoom)*(1-on/%d)" % n
    elif direction == "right":
        cx = "(iw-iw/zoom)*on/%d" % n
    elif direction == "up":
        cy = "(ih-ih/zoom)*(1-on/%d)" % n
    elif direction == "down":
        cy = "(ih-ih/zoom)*on/%d" % n
    elif direction not in ("in", "out", "none"):
        raise EditError("direction must be in, out, left, right, up, down or none.")
    f = "[0:v]scale=%d:%d:flags=lanczos,zoompan=z='%s':x='%s':y='%s':d=1:s=%dx%d:fps=%s[v]" % (w * 2, h * 2, zx, cx, cy, w, h, fps)
    dst = _new("zoom")
    _encode(["-i", src], f, dst, _bitrate(w, h, fps), info["audio"], ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    return _finish(dst, "zoom", [src], {"zoom": z, "direction": direction})


def smooth_motion(v, fps=48):
    """Makes in-between frames (24 -> 48/60 fps) on the processor (ffmpeg motion interpolation). Slow on long clips."""
    fps = int(fps)
    if fps not in (30, 48, 50, 60):
        raise EditError("fps must be 30, 48, 50 or 60.")
    src = resolve(v)
    info = probe(src)
    dst = _new("smooth")
    _encode(["-i", src], "[0:v]minterpolate=fps=%d:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1[v]" % fps, dst,
            _bitrate(info["width"], info["height"], fps), info["audio"], ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    return _finish(dst, "smooth", [src], {"fps": fps})


def upscale_simple(v, height=1080):
    src = resolve(v)
    info = probe(src)
    h = int(height) // 2 * 2
    w = int(round(info["width"] * h / info["height"] / 2)) * 2
    dst = _new("upscale")
    _encode(["-i", src], "[0:v]scale=%d:%d:flags=lanczos,unsharp=5:5:0.6[v]" % (w, h) if " unsharp " in ff()["filters"]
            else "[0:v]scale=%d:%d:flags=lanczos[v]" % (w, h), dst, _bitrate(w, h, info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    return _finish(dst, "upscale", [src], {"height": h, "method": "simple"})


# ---------- joining ----------
def join(videos, transition="cut", seconds=0.5, size=None):
    """Clips into one video. transition: cut | fade (through black) | crossfade. All clips get the first clip's size
    (others are fitted with bars) and frame rate."""
    if len(videos) < 2:
        raise EditError("Give at least 2 videos to join.")
    if transition not in ("cut", "fade", "crossfade"):
        raise EditError("transition must be cut, fade or crossfade.")
    srcs = [resolve(v) for v in videos]
    infos = [probe(s) for s in srcs]
    W, H = size or (infos[0]["width"], infos[0]["height"])
    fps = infos[0]["fps"]
    td = max(0.1, min(float(seconds), min(i["duration"] for i in infos) / 2 - 0.05)) if transition != "cut" else 0
    args, parts, extra_inputs = [], [], 0
    for s in srcs:
        args += ["-i", s]
    n = len(srcs)
    for i, (s, inf) in enumerate(zip(srcs, infos)):
        parts.append("[%d:v]scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=%s,format=yuv420p[v%d]"
                     % (i, W, H, W, H, fps, i))
        if inf["audio"]:
            parts.append("[%d:a]aresample=44100,aformat=channel_layouts=stereo[a%d]" % (i, i))
        else:
            args += ["-f", "lavfi", "-t", "%.3f" % inf["duration"], "-i", "anullsrc=r=44100:cl=stereo"]
            parts.append("[%d:a]anull[a%d]" % (n + extra_inputs, i))
            extra_inputs += 1
    if transition == "cut":
        parts.append("".join("[v%d][a%d]" % (i, i) for i in range(n)) + "concat=n=%d:v=1:a=1[v][a]" % n)
    else:
        kind = "fade" if transition == "crossfade" else "fadeblack"
        prev_v, prev_a, offset = "[v0]", "[a0]", 0.0
        for i in range(1, n):
            offset += infos[i - 1]["duration"] - td
            ov, oa = ("[v]", "[a]") if i == n - 1 else ("[vx%d]" % i, "[ax%d]" % i)
            parts.append("%s[v%d]xfade=transition=%s:duration=%.3f:offset=%.3f%s" % (prev_v, i, kind, td, offset, ov))
            parts.append("%s[a%d]acrossfade=d=%.3f%s" % (prev_a, i, td, oa))
            prev_v, prev_a = ov, oa
    dst = _new("join")
    _encode(args, ";".join(parts), dst, _bitrate(W, H, fps), True, ["-map", "[v]", "-map", "[a]"])
    return _finish(dst, "join", srcs, {"transition": transition, "seconds": td})


# ---------- shape and export ----------
def reformat(v, shape="9:16", mode="blur"):
    """9:16 (Shorts, Reels, TikTok) | 1:1 | 4:5 | 16:9. mode: blur (blurred copy behind, nothing cut off) | crop (fill, cuts the sides) | pad (black bars)."""
    if shape not in SHAPES:
        raise EditError("shape must be one of " + ", ".join(SHAPES))
    if mode not in ("blur", "crop", "pad"):
        raise EditError("mode must be blur, crop or pad.")
    src = resolve(v)
    info = probe(src)
    W, H = SHAPES[shape]
    if mode == "crop":
        f = "[0:v]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,setsar=1[v]" % (W, H, W, H)
    elif mode == "pad":
        f = "[0:v]scale=%d:%d:force_original_aspect_ratio=decrease,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1[v]" % (W, H, W, H)
    else:
        f = ("[0:v]split[a][b];[a]scale=%d:%d:force_original_aspect_ratio=increase,crop=%d:%d,gblur=sigma=40,lutyuv=y='val*0.7'[bg];"
             "[b]scale=%d:%d:force_original_aspect_ratio=decrease[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[v]") % (W, H, W, H, W, H)
    dst = _new("reformat")
    _encode(["-i", src], f, dst, _bitrate(W, H, info["fps"]), info["audio"], ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    return _finish(dst, "reformat", [src], {"shape": shape, "mode": mode})


def export(v, platform="shorts", max_mb=None):
    """Ready to post: the platform's shape (blurred background if needed), H.264 + AAC, size under the limit."""
    if platform not in PLATFORMS:
        raise EditError("platform must be one of " + ", ".join(PLATFORMS))
    shape, lim, max_s = PLATFORMS[platform]
    lim = float(max_mb or lim)
    src = resolve(v)
    info = probe(src)
    if info["duration"] > max_s:
        raise EditError("The video is %.0f s; %s takes at most %d s. Trim it first." % (info["duration"], platform, max_s))
    W, H = SHAPES[shape]
    if abs(info["width"] / info["height"] - W / H) > 0.02:
        src = resolve(reformat(src, shape, "blur")["path"])
        info = probe(src)
    br = int(min(_bitrate(W, H, info["fps"]), (lim * 8e6 * 0.9) / max(0.5, info["duration"]) - 160e3))
    if br < 300e3:
        raise EditError("%.0f MB is too small for %.0f s of video. Allow more MB or trim it." % (lim, info["duration"]))
    dst = _new("export_" + platform)
    vf = "[0:v]scale=%d:%d,setsar=1[v]" % (W, H)
    if info["audio"]:  # loudness evened out to what social apps play (-14 LUFS), so it isn't quiet next to other posts
        _encode(["-i", src], vf + ";[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=44100[a]", dst, br, True, ["-map", "[v]", "-map", "[a]"])
    else:  # platforms like a sound track; add silence
        _encode(["-i", src, "-f", "lavfi", "-t", "%.3f" % info["duration"], "-i", "anullsrc=r=44100:cl=stereo"], vf, dst, br, True,
                ["-map", "[v]", "-map", "1:a", "-shortest"])
    r = _finish(dst, "export", [src], {"platform": platform, "max_mb": lim})
    if r["mb"] > lim:
        r["warning"] = "Came out %.1f MB, over %.0f MB. Export again with a lower max_mb." % (r["mb"], lim)
    return r


def thumbnail(v, at=None):
    src = resolve(v)
    info = probe(src)
    t = info["duration"] / 2 if at is None else max(0, min(float(at), info["duration"] - 0.05))
    dst = _new("thumb")[:-4] + ".jpg"
    _run(["-ss", "%.3f" % t, "-i", src, "-frames:v", "1", "-q:v", "2", dst], "Thumbnail")
    return {"image": os.path.basename(dst), "path": dst, "at": round(t, 2)}


def make_gif(v, fps=12, width=480):
    src = resolve(v)
    dst = _new("gif")[:-4] + ".gif"
    f = "[0:v]fps=%d,scale=%d:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer" % (int(fps), int(width))
    _run(["-i", src, "-filter_complex", f, dst], "GIF")
    return {"gif": os.path.basename(dst), "path": dst, "mb": round(os.path.getsize(dst) / 1e6, 2)}


def extract_audio(v):
    src = resolve(v)
    if not probe(src)["audio"]:
        raise EditError("This video has no sound.")
    dst = _new("audio")[:-4] + ".m4a"
    _run(["-i", src, "-vn", "-c:a", "aac", "-b:a", "192k", dst], "Audio")
    return {"audio": os.path.basename(dst), "path": dst}


def remove_audio(v):
    src = resolve(v)
    info = probe(src)
    dst = _new("mute")
    _encode(["-i", src], None, dst, _bitrate(info["width"], info["height"], info["fps"]), False)
    return _finish(dst, "mute", [src], {})


# ---------- sound ----------
def add_music(v, music, volume=0.25, fade_s=1.0, duck=True):
    """Background music, looped or cut to the video's length, faded in/out. If the video already has a voice,
    the music gets quieter while the voice speaks (duck)."""
    src, mus = resolve(v), resolve_any(music, "music file")
    info = probe(src)
    d = info["duration"]
    m = "[1:a]volume=%.3f,atrim=0:%.3f,afade=t=in:st=0:d=%.2f,afade=t=out:st=%.3f:d=%.2f,aformat=channel_layouts=stereo[m]" % (
        float(volume), d, fade_s, max(0, d - fade_s), fade_s)
    if info["audio"]:
        if duck:
            f = m + ";[0:a]aformat=channel_layouts=stereo,asplit[voice][key];[m][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[md];" \
                    "[voice][md]amix=inputs=2:duration=first:normalize=0[a]"
        else:
            f = m + ";[0:a]aformat=channel_layouts=stereo[voice];[voice][m]amix=inputs=2:duration=first:normalize=0[a]"
    else:
        f = m.replace("[m]", "[a]")
    dst = _new("music")
    _encode(["-i", src, "-stream_loop", "-1", "-i", mus], f, dst, _bitrate(info["width"], info["height"], info["fps"]), True,
            ["-map", "0:v", "-map", "[a]", "-t", "%.3f" % d])
    return _finish(dst, "music", [src], {"music": os.path.basename(mus), "volume": volume, "duck": duck})


def add_voice(v, audio, volume=1.0, start=0.0, keep_original=False):
    """Puts a voice line (or any sound) on a clip. If the sound is longer than the clip, the last frame is held."""
    src, aud = resolve(v), resolve_any(audio, "audio file")
    info, ai = probe(src), probe(aud)
    need = float(start) + ai["duration"]
    hold = max(0.0, need - info["duration"] + 0.15)
    vf = "[0:v]tpad=stop_mode=clone:stop_duration=%.3f[v]" % hold if hold > 0 else "[0:v]null[v]"
    delay = int(float(start) * 1000)
    a = "[1:a]volume=%.3f,adelay=%d|%d,aformat=channel_layouts=stereo[vo]" % (float(volume), delay, delay)
    if keep_original and info["audio"]:
        a += ";[0:a]aformat=channel_layouts=stereo[o];[o][vo]amix=inputs=2:duration=longest:normalize=0[a]"
    else:
        a = a.replace("[vo]", "[a]")
    dst = _new("voice")
    _encode(["-i", src, "-i", aud], vf + ";" + a, dst, _bitrate(info["width"], info["height"], info["fps"]), True,
            ["-map", "[v]", "-map", "[a]", "-t", "%.3f" % max(info["duration"], need + 0.15)])
    words = os.path.splitext(aud)[0] + ".words.json"
    extra = {"voice_words": words, "voice_start": float(start)} if os.path.exists(words) else None
    return _finish(dst, "voice", [src], {"audio": os.path.basename(aud), "start": start}, extra)


EDGE_VOICES = {"andrew": "en-US-AndrewMultilingualNeural", "emma": "en-US-EmmaMultilingualNeural",
               "ava": "en-US-AvaMultilingualNeural", "brian": "en-US-BrianMultilingualNeural",
               "es": "es-ES-AlvaroNeural", "es-f": "es-ES-ElviraNeural", "pt": "pt-PT-DuarteNeural", "pt-f": "pt-PT-RaquelNeural"}


def make_voice(text, voice="andrew", engine="edge", sample=None, language="en", rate="+0%"):
    """A voice line from text. engine "edge": Microsoft's free online voices (needs internet; edge-tts), with word
    timings for captions. engine "local": Chatterbox on this PC (offline; clones the voice in `sample`, a 5-15 s
    clean recording; slow on the processor, first use downloads about 3 GB)."""
    text = (text or "").strip()
    if not text:
        raise EditError("No text for the voice.")
    if len(text) > 3000:
        raise EditError("Text is too long (max 3000 characters); split it into lines.")
    os.makedirs(VOICE, exist_ok=True)
    name = "vo_%s_%s" % (time.strftime("%m%d_%H%M%S"), uuid.uuid4().hex[:4])
    if engine == "edge":
        return _edge_voice(text, EDGE_VOICES.get(voice, voice), rate, os.path.join(VOICE, name))
    if engine == "local":
        return _chatterbox(text, sample, language, os.path.join(VOICE, name))
    raise EditError('engine must be "edge" or "local".')


def _edge_voice(text, voice, rate, base):
    try:
        import asyncio
        import edge_tts
    except ImportError:
        raise EditError("edge-tts is not installed (pip install edge-tts), or use engine=\"local\".")

    async def go():
        com = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
        words, audio = [], bytearray()
        async for ch in com.stream():
            if ch["type"] == "audio":
                audio += ch["data"]
            elif ch["type"] == "WordBoundary":
                words.append({"word": ch["text"], "start": ch["offset"] / 1e7, "end": (ch["offset"] + ch["duration"]) / 1e7})
        return bytes(audio), words
    try:
        audio, words = asyncio.run(go())
    except Exception as e:
        raise EditError("The online voice service didn't answer (%s). Check the internet, or use engine=\"local\"." % str(e)[:200])
    words = _punctuate(words, text)  # the service's words come without commas and full stops; captions break on them
    open(base + ".mp3", "wb").write(audio)
    json.dump(words, open(base + ".words.json", "w"), indent=0)
    return {"audio": os.path.basename(base) + ".mp3", "path": base + ".mp3", "seconds": round(probe(base + ".mp3")["duration"], 2),
            "voice": voice, "words": len(words), "note": "Word timings saved: add_captions(..., text=\"auto\") uses them."}


def _punctuate(words, text):
    """Gives each timed word its spelling from the original text (with punctuation), matching in order."""
    import re
    toks, j = text.split(), 0
    norm = lambda s: re.sub(r"[^\w']", "", s.lower())
    for w in words:
        for k in range(j, min(j + 4, len(toks))):
            if norm(toks[k]) == norm(w["word"]):
                w["word"], j = toks[k], k + 1
                break
    return words


def _chatterbox(text, sample, language, base):
    py = os.environ.get("CHATTERBOX_PYTHON") or r"C:\LLM\chatterbox\venv\Scripts\python.exe"
    script = os.environ.get("CHATTERBOX_SCRIPT") or os.path.join(HERE, "..", "local-voice", "clone.py")
    if not (os.path.exists(py) and os.path.exists(script)):
        raise EditError("Chatterbox is not installed on this PC (set CHATTERBOX_PYTHON and CHATTERBOX_SCRIPT), or use engine=\"edge\".")
    if not sample:
        raise EditError("engine \"local\" clones a voice: give sample = a 5-15 s recording of the voice.")
    ref = resolve_any(sample, "voice sample")
    raw = base + ".raw.wav"
    r = subprocess.run([py, script, ref, language, text, raw, "--cpu"], capture_output=True, text=True, timeout=3600)
    if r.returncode or not os.path.exists(raw):
        raise EditError("Chatterbox failed: " + (r.stderr or r.stdout)[-500:])
    _run(["-i", raw, "-c:a", "pcm_s16le", base + ".wav"], "Voice")  # Chatterbox writes float WAV; most players want 16-bit
    os.remove(raw)
    return {"audio": os.path.basename(base) + ".wav", "path": base + ".wav", "seconds": round(probe(base + ".wav")["duration"], 2),
            "note": "Local voice (Chatterbox, carries Resemble's inaudible watermark). No word timings: captions are spread evenly."}


# ---------- text on video (drawn with Pillow) ----------
FONTS = {"bold": ["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"], "impact": ["impact.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
         "clean": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"]}


def _font(style, size):
    from PIL import ImageFont
    for name in ([os.environ["CAPTION_FONT"]] if os.environ.get("CAPTION_FONT") else []) + FONTS.get(style, FONTS["bold"]):
        for d in ("", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"), "/usr/share/fonts/truetype/dejavu"):
            p = os.path.join(d, name) if d else name
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _text_png(text, W, size, style="bold", box=False, color=(255, 255, 255), max_frac=0.88):
    """A transparent picture of the text, wrapped to fit max_frac of the width, white with a dark outline."""
    from PIL import Image, ImageDraw
    font = _font("impact" if style == "impact" else ("clean" if style == "clean" else "bold"), size)
    d0 = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if d0.textlength(t, font=font) <= W * max_frac or not cur:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    stroke = 0 if box else max(2, size // 12)
    lh = int(size * 1.18)
    tw = int(max(d0.textlength(l, font=font) for l in lines)) + stroke * 2 + (size if box else 0)
    th = lh * len(lines) + stroke * 2 + (size // 2 if box else 0)
    im = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    if box:
        d.rounded_rectangle([0, 0, tw - 1, th - 1], radius=size // 3, fill=(0, 0, 0, 170))
    for i, l in enumerate(lines):
        x = (tw - d.textlength(l, font=font)) / 2
        y = (size // 4 if box else stroke) + i * lh
        d.text((x, y), l, font=font, fill=color + (255,), stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    os.makedirs(TMP, exist_ok=True)
    p = os.path.join(TMP, "txt_%s.png" % uuid.uuid4().hex[:8])
    im.save(p)
    return p


def _ypos(position):
    return {"top": "H*0.10", "center": "(H-h)/2", "bottom": "H*0.80-h/2", "lower": "H*0.70-h/2"}.get(position, "H*0.80-h/2")


def _overlay_pngs(src, info, items, op, params, extra_meta=None):
    """items: [(png, x_expr, y_expr, start, end)]; overlays them in one pass."""
    args, parts, prev = ["-i", src], [], "[0:v]"
    for i, (png, x, y, a, b) in enumerate(items):
        args += ["-i", png]
        out = "[v]" if i == len(items) - 1 else "[o%d]" % i
        en = ":enable='between(t,%.3f,%.3f)'" % (a, b) if b is not None else ""
        parts.append("%s[%d:v]overlay=x=%s:y=%s%s%s" % (prev, i + 1, x, y, en, out))
        prev = out
    dst = _new(op)
    _encode(args, ";".join(parts), dst, _bitrate(info["width"], info["height"], info["fps"]), info["audio"],
            ["-map", "[v]"] + (["-map", "0:a"] if info["audio"] else []))
    for png, *_ in items:
        try:
            os.remove(png)
        except OSError:
            pass
    return _finish(dst, op, [src], params, extra_meta)


def add_text(v, text, position="top", start=0.0, end=None, size=None, style="bold", box=False):
    src = resolve(v)
    info = probe(src)
    size = int(size or min(info["width"], info["height"]) * 0.08)
    png = _text_png(text, info["width"], size, style, box)
    end = info["duration"] if end is None else float(end)
    return _overlay_pngs(src, info, [(png, "(W-w)/2", _ypos(position), float(start), end)], "text",
                         {"text": text, "position": position, "start": start, "end": end})


def add_logo(v, image, position="top-right", scale=0.15, opacity=0.9):
    from PIL import Image
    src, img = resolve(v), resolve_any(image, "logo picture")
    info = probe(src)
    im = Image.open(img).convert("RGBA")
    w = max(16, int(info["width"] * float(scale)))
    im = im.resize((w, max(1, int(im.height * w / im.width))))
    if opacity < 1:
        im.putalpha(im.getchannel("A").point(lambda a: int(a * float(opacity))))
    os.makedirs(TMP, exist_ok=True)
    png = os.path.join(TMP, "logo_%s.png" % uuid.uuid4().hex[:8])
    im.save(png)
    m = "W*0.03"
    pos = {"top-right": ("W-w-%s" % m, m), "top-left": (m, m), "bottom-right": ("W-w-%s" % m, "H-h-%s" % m),
           "bottom-left": (m, "H-h-%s" % m), "center": ("(W-w)/2", "(H-h)/2")}
    if position not in pos:
        raise EditError("position must be " + ", ".join(pos))
    x, y = pos[position]
    return _overlay_pngs(src, info, [(png, x, y, 0, None)], "logo", {"logo": os.path.basename(img), "position": position})


def _chunks_from_words(words, offset, max_words=4, max_gap=0.6):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap or cur[-1]["word"][-1:] in ".!?,;:"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return [(" ".join(x["word"] for x in c), c[0]["start"] + offset, c[-1]["end"] + offset + 0.05) for c in out]


def add_captions(v, text="auto", style="bold", position="bottom", size=None, words_per_caption=4, highlight=False):
    """Burned-in captions. text="auto": timed to the voice line added with add_voice (from make_voice with the
    edge engine). Or give the text: it is split into short captions spread over the clip ("|" forces a break).
    Or a list of {"text", "start", "end"}. style: bold | impact | clean | box."""
    src = resolve(v)
    info = probe(src)
    meta = _meta(src)
    caps = []
    if isinstance(text, list):
        caps = [(c["text"], float(c["start"]), float(c["end"])) for c in text]
    elif text == "auto":
        wf = meta.get("voice_words")
        if not wf or not os.path.exists(wf):
            raise EditError("No voice timings for this video. Use add_voice with a make_voice (edge) line first, or give the text.")
        caps = _chunks_from_words(json.load(open(wf)), float(meta.get("voice_start", 0)), int(words_per_caption))
    else:
        if "|" in text:
            pieces = [p.strip() for p in text.split("|") if p.strip()]
        else:
            ws = text.split()
            pieces = [" ".join(ws[i:i + int(words_per_caption)]) for i in range(0, len(ws), int(words_per_caption))]
        total = sum(len(p) for p in pieces) or 1
        t, d = 0.0, info["duration"]
        for p in pieces:
            span = d * len(p) / total
            caps.append((p, t, t + span))
            t += span
    if not caps:
        raise EditError("No captions to add.")
    if len(caps) > 120:
        raise EditError("Too many captions (%d); max 120 per call." % len(caps))
    size = int(size or min(info["width"], info["height"]) * (0.085 if info["height"] > info["width"] else 0.075))
    color = (255, 230, 60) if highlight else (255, 255, 255)
    items = [(_text_png(c, info["width"], size, style, style == "box", color), "(W-w)/2", _ypos(position), a, b) for c, a, b in caps]
    return _overlay_pngs(src, info, items, "captions", {"captions": len(caps), "style": style, "auto": text == "auto"})


def card(text="", image=None, seconds=2.0, size=(1080, 1920), bg=(12, 12, 16), style="bold"):
    """A still card (title, end card): text and/or a picture on a dark background, as a short video with silence."""
    from PIL import Image
    W, H = size
    im = Image.new("RGB", (W, H), bg)
    if image:
        pic = Image.open(resolve_any(image, "card picture")).convert("RGBA")
        pic.thumbnail((int(W * 0.7), int(H * (0.45 if text else 0.7))))
        im.paste(pic, ((W - pic.width) // 2, int(H * (0.18 if text else 0.15)) if text else (H - pic.height) // 2), pic)
    if text:
        t = Image.open(_text_png(text, W, int(min(W, H) * 0.09), style))
        y = int(H * 0.68) - t.height // 2 if image else (H - t.height) // 2
        im.paste(t, ((W - t.width) // 2, y), t)
    os.makedirs(TMP, exist_ok=True)
    png = os.path.join(TMP, "card_%s.png" % uuid.uuid4().hex[:8])
    im.save(png)
    dst = _new("card")
    _encode(["-loop", "1", "-framerate", "24", "-t", "%.3f" % float(seconds), "-i", png, "-f", "lavfi", "-t", "%.3f" % float(seconds),
             "-i", "anullsrc=r=44100:cl=stereo"], "[0:v]fade=t=in:st=0:d=0.3,format=yuv420p[v]", dst, _bitrate(W, H, 24), True,
            ["-map", "[v]", "-map", "1:a", "-shortest"])
    os.remove(png)
    return _finish(dst, "card", [], {"text": text, "seconds": seconds})


def title_card(v, text="", image=None, seconds=2.0, transition="fade"):
    src = resolve(v)
    i = probe(src)
    c = card(text, image, seconds, (i["width"], i["height"]))
    return join([c["path"], src], transition, 0.4)


def end_card(v, text="Follow for more", image=None, seconds=2.5, transition="fade"):
    src = resolve(v)
    i = probe(src)
    c = card(text, image, seconds, (i["width"], i["height"]))
    return join([src, c["path"]], transition, 0.4)


def last_frame(v, dst_png):
    src = resolve(v)
    info = probe(src)
    _run(["-sseof", "-0.08", "-i", src, "-frames:v", "1", "-update", "1", dst_png], "Last frame")
    if not os.path.exists(dst_png):
        _run(["-ss", "%.3f" % max(0, info["duration"] - 0.1), "-i", src, "-frames:v", "1", dst_png], "Last frame")
    return dst_png
