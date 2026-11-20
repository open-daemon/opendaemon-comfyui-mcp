"""AI Video Pipeline - test UI (later a tab in OpenDaemon).
Serves index.html on http://127.0.0.1:8190 and talks to ComfyUI on 127.0.0.1:8188.
Every LTX setting is exposed so a human (or later an agent over MCP) can tune it.

Run: python server.py
"""
import base64
import glob
import json
import os
import random
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # embedded Pythons don't add the script's folder
if os.environ.get("COMFY_MCP_PACKAGES"):  # packages shipped next to us (edge-tts); embedded Pythons ignore PYTHONPATH
    import site
    sys.path.insert(1, os.environ["COMFY_MCP_PACKAGES"])
    site.addsitedir(os.environ["COMFY_MCP_PACKAGES"])
import paths  # where ComfyUI lives (COMFYUI_DIR / COMFYUI_URL)
COMFY = paths.COMFY
URL = paths.URL
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.OUT
DATA = paths.DATA
PORT = int(os.environ.get("PIPELINE_PORT", 8190))
JOBS = {}  # id -> dict(status, started, ended, settings, video, error, prompt_id)


import models_mgr  # downloads, "add your own model", the file check


def friendly_error(raw):
    """ComfyUI's errors are for developers. Say what happened first; keep the raw text below for us."""
    t = raw.lower()
    if "out of memory" in t or "outofmemory" in t or "cuda error: out of memory" in t or "allocation" in t and "failed" in t:
        why = ("Not enough graphics card memory for this model at this size. Try Quality: Draft or Normal, a shorter "
               "length, or a smaller version of the model (Q3 or Q4 instead of Q8).")
    elif "value_not_in_list" in t or "not in list" in t:
        why = "A model file this video needs is missing. Open Models and download it (it may have been moved or deleted)."
    elif "size mismatch" in t or "error(s) in loading state_dict" in t or "unexpected key" in t or "missing key" in t:
        why = "This model file doesn't fit the model type it was added as, so it couldn't be loaded. Remove it in Models and add it again."
    elif "no ffmpeg" in t or "ffmpeg" in t and "fail" in t:
        why = "The video frames were made, but turning them into an mp4 failed."
    elif "connection refused" in t or "urlopen error" in t:
        why = "ComfyUI stopped or isn't running. Start it again, then render again."
    else:
        return raw
    return why + "\n\nDetails: " + raw[-600:]


CLIENT_ID = str(uuid.uuid4())
PROGRESS = {}  # prompt_id -> dict(value, max, pass_no): live sampler steps from ComfyUI's websocket


def progress_listener():
    """Minimal stdlib websocket client for ws://127.0.0.1:8188/ws. Keeps PROGRESS current; reconnects forever."""
    import socket, struct
    while True:
        try:
            s = socket.create_connection((paths.HOST, paths.PORT), timeout=60)
            key = base64.b64encode(os.urandom(16)).decode()
            s.sendall(("GET /ws?clientId=%s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\n"
                       "Connection: Upgrade\r\nSec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n" % (CLIENT_ID, paths.HOST, paths.PORT, key)).encode())
            buf = b""
            while b"\r\n\r\n" not in buf:
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("handshake")
                buf += chunk
            buf = buf.split(b"\r\n\r\n", 1)[1]

            def need(n):
                nonlocal buf
                while len(buf) < n:
                    chunk = s.recv(65536)
                    if not chunk:
                        raise ConnectionError("closed")
                    buf += chunk
                out, buf = buf[:n], buf[n:]
                return out

            while True:
                b1, b2 = need(2)
                op, ln = b1 & 0x0F, b2 & 0x7F
                if ln == 126:
                    ln = struct.unpack(">H", need(2))[0]
                elif ln == 127:
                    ln = struct.unpack(">Q", need(8))[0]
                data = need(ln)
                if op == 8:
                    raise ConnectionError("close frame")
                if op == 9:  # ping -> masked pong
                    s.sendall(bytes([0x8A, 0x80 | len(data)]) + b"\0\0\0\0" + data)
                if op != 1:
                    continue
                m = json.loads(data)
                d = m.get("data") or {}
                pid = d.get("prompt_id")
                if m.get("type") == "progress" and pid:
                    p = PROGRESS.setdefault(pid, {"value": 0, "max": 1, "pass_no": 1})
                    if d["value"] < p["value"]:  # sampler restarted = second pass (refine)
                        p["pass_no"] += 1
                    p.update(value=d["value"], max=d["max"])
        except Exception:
            time.sleep(3)


threading.Thread(target=progress_listener, daemon=True).start()


def comfy(path, body=None):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    # /interrupt, /free and /queue (delete) answer with an empty body: that is success, not bad JSON.
    return json.loads(data) if data.strip() else {}


def options():
    """Lists the UI needs, read live from ComfyUI so new models show up by themselves."""
    def node_list(node, field):
        try:
            info = comfy("/object_info/" + node)[node]["input"]
            spec = info.get("required", {}).get(field) or info.get("optional", {}).get(field)
            v = spec[0]
            return v if isinstance(v, list) else (spec[1].get("options") if len(spec) > 1 else [])
        except Exception:
            return []
    return {
        "checkpoints": node_list("CheckpointLoaderSimple", "ckpt_name"),
        "text_encoders": node_list("CLIPLoader", "clip_name"),
        "samplers": sorted(set(node_list("KSamplerSelect", "sampler_name")) | set(node_list("KSampler", "sampler_name"))),
        "upscalers": node_list("LatentUpscaleModelLoader", "model_name"),
        "unets": node_list("UnetLoaderGGUF", "unet_name"),
        "diffusion": node_list("UNETLoader", "unet_name"),
        "images": sorted(f for f in os.listdir(os.path.join(COMFY, "input"))
                         if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))),
    }


def workflow(s, outdir):
    """s = settings dict from the page. family picks the model type (see families.py)."""
    import families
    fam = s.get("family") or "ltx"
    if fam == "vace":
        return families.vace(s, outdir)
    wf = workflow_ltx(s, outdir)
    return families.ltx13(wf, s) if fam == "ltx13" else wf


def workflow_ltx(s, outdir):
    """The LTX graph (same as local-video/animate.py), every knob open.
    No picture (s["image"] empty) = text to video: an empty video latent instead of the picture's."""
    seed = int(s["seed"])
    pic = bool(s.get("image"))
    pos, neg, lat = (["7", 0], ["7", 1], ["7", 2]) if pic else (["3", 0], ["4", 0], ["7", 0])
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": s["checkpoint"]}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": s["text_encoder"], "type": "ltxv", "device": "default"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": s["prompt"], "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": s["negative"], "clip": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": s["image"]}},
        "6": {"class_type": "LTXVPreprocess", "inputs": {"image": ["5", 0], "img_compression": int(s["comp"])}},
        "7": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "vae": ["1", 2], "image": ["6", 0],
            "width": int(s["width"]), "height": int(s["height"]), "length": int(s["frames"]),
            "batch_size": 1, "strength": float(s["strength"])}},
        "8": {"class_type": "LTXVConditioning", "inputs": {"positive": pos, "negative": neg,
                                                           "frame_rate": float(s["cfps"] or 0) or float(s["fps"])}},
        "9": {"class_type": "ModelSamplingLTXV", "inputs": {"model": ["1", 0], "max_shift": float(s["max_shift"]),
                                                            "base_shift": float(s["base_shift"]), "latent": lat}},
        "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": s["sampler"]}},
        "12": {"class_type": "SamplerCustom", "inputs": {
            "model": ["9", 0], "add_noise": True, "noise_seed": seed, "cfg": float(s["cfg"]),
            "positive": ["8", 0], "negative": ["8", 1], "sampler": ["11", 0], "sigmas": ["10", 0],
            "latent_image": lat}},
        "13": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["12", 0], "vae": ["1", 2], "tile_size": int(s["tile"]), "overlap": 64,
            "temporal_size": 32, "temporal_overlap": 8}},
        "14": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": outdir + "/f"}},
    }
    if s["schedule"] == "manual":
        wf["10"] = {"class_type": "ManualSigmas", "inputs": {"sigmas": s["sigmas"]}}
    else:
        wf["10"] = {"class_type": "LTXVScheduler", "inputs": {
            "steps": int(s["steps"]), "max_shift": float(s["max_shift"]), "base_shift": float(s["base_shift"]),
            "stretch": bool(s["stretch"]), "terminal": float(s["terminal"]), "latent": lat}}
    if s.get("refine"):
        wf.update({
            "20": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": s["upscaler"]}},
            "21": {"class_type": "LTXVLatentUpsampler", "inputs": {"samples": ["12", 0], "upscale_model": ["20", 0],
                                                                   "vae": ["1", 2]}},
            "22": {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": ["1", 2], "image": ["6", 0],
                                                                     "latent": ["21", 0], "strength": 1.0,
                                                                     "bypass": False}},
            "23": {"class_type": "ManualSigmas", "inputs": {"sigmas": s["refine_sigmas"]}},
            "24": {"class_type": "SamplerCustom", "inputs": {
                "model": ["9", 0], "add_noise": True, "noise_seed": seed, "cfg": float(s["cfg"]),
                "positive": ["8", 0], "negative": ["8", 1], "sampler": ["11", 0], "sigmas": ["23", 0],
                "latent_image": ["22", 0]}},
        })
        wf["13"]["inputs"]["samples"] = ["24", 0]
        if not pic:  # nothing to pin the first frame to: refine the upscaled latent as it is
            del wf["22"]
            wf["24"]["inputs"]["latent_image"] = ["21", 0]
    if not pic:
        for k in ("5", "6"):
            del wf[k]
        wf["7"] = {"class_type": "EmptyLTXVLatentVideo", "inputs": {
            "width": int(s["width"]), "height": int(s["height"]), "length": int(s["frames"]), "batch_size": 1}}
    return wf


def to_mp4(frames_dir, fps, dst, bitrate):
    # CapCut's ffmpeg (hardware h264_mf) first; it once failed to start (0xC0000142), so fall back to
    # Remotion's ffmpeg with libx264 instead of losing a finished render.
    inp = ["-v", "error", "-y", "-framerate", str(fps), "-start_number", "1", "-i", os.path.join(frames_dir, "f_%05d_.png")]
    tries = []
    for f in paths.ffmpeg_candidates():  # CapCut's has only hardware encoders; the others usually have libx264
        encs = [["-c:v", "h264_mf", "-b:v", str(bitrate)]] if "CapCut" in f else [
            ["-c:v", "libx264", "-crf", "16"], ["-c:v", "h264_nvenc", "-b:v", str(bitrate)], ["-c:v", "h264_mf", "-b:v", str(bitrate)]]
        tries += [(f, e) for e in encs]
    err = None
    for ff, enc in tries:
        if not os.path.exists(ff):
            continue
        try:
            subprocess.run([ff] + inp + enc + ["-pix_fmt", "yuv420p", dst], check=True)
            return
        except Exception as e:
            err = e
    raise err or RuntimeError("no ffmpeg found")


# ---------- our own queue (phase 2): priority, pause, clear ----------
# Jobs wait HERE, not in ComfyUI, so a batch never blocks the user's own render: "user" jobs (the page, a single
# render) go before "batch" jobs. Within a priority, oldest first; "front" moves a job to the head of its priority.
QLOCK = threading.Lock()
QSTATE = {"paused": False}
PRIO = {"user": 0, "batch": 1}


def waiting():
    return sorted([(PRIO.get(j.get("priority"), 0), j.get("order", j["started"]), k) for k, j in JOBS.items()
                   if j["status"] == "queued" and not j.get("sent")])


def my_turn(jid):
    with QLOCK:
        if QSTATE["paused"]:
            return False
        if any(j.get("sent") and j["status"] in ("queued", "rendering") for j in JOBS.values()):
            return False
        w = waiting()
        if w and w[0][2] == jid:
            JOBS[jid]["sent"] = True
            return True
        return False


_GPU = []


def GPU_NAME():
    if not _GPU:
        try:
            _GPU.append(comfy("/system_stats")["devices"][0]["name"].split(":")[1].strip().split(" : ")[0])
        except Exception:
            return None
    return _GPU[0]


def minutes_for(s):
    """Rough minutes for one render: our guess for a 4 GB card, corrected by how long renders really took on THIS PC."""
    if s.get("workflow"):  # your own workflow: the median of its last runs on this PC, else a guess
        try:
            t = sorted(json.load(open(WF_TIMES)).get(s["workflow"]) or [])
        except Exception:
            t = []
        return (max(1, round(t[len(t) // 2] / 60, 1)) if t else 3), len(t)
    import auto
    guess = auto.guess_minutes(s)
    ratios = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json")), key=os.path.getmtime, reverse=True)[:60]:
        try:
            h = json.load(open(p))
        except Exception:
            continue
        if h.get("render_seconds") and h.get("family", "ltx") == s.get("family", "ltx") and int(h.get("steps", 8)) == int(s.get("steps", 8)):
            ratios.append(h["render_seconds"] / 60 / max(0.5, auto.guess_minutes(h)))
    if len(ratios) >= 2:
        ratios.sort()
        guess *= ratios[len(ratios) // 2]
    return max(1, round(guess, 1)), len(ratios)


def queue_view():
    now, ahead, out = time.time(), 0.0, []
    for k, j in sorted(JOBS.items(), key=lambda x: (x[1]["status"] != "rendering", not x[1].get("sent"),
                                                    PRIO.get(x[1].get("priority"), 0), x[1].get("order", x[1]["started"]))):
        if j["status"] not in ("queued", "rendering"):
            continue
        m = minutes_for(j.get("settings") or {})[0] if j.get("settings") else 5
        left = m - (now - j["gpu_started"]) / 60 if j.get("gpu_started") else m
        out.append({"id": k, "status": j["status"] if j.get("sent") else "waiting", "priority": j.get("priority", "user"),
                    "settings": (j.get("settings") or {}).get("easy"), "batch": j.get("batch"),
                    "starts_in_min": round(ahead) if not j.get("gpu_started") else 0, "minutes": m})
        ahead += max(0.5, left)
    return {"paused": QSTATE["paused"], "jobs": out, "all_done_in_min": round(ahead)}


def queue_action(a, jid=None, priority=None):
    if a == "pause":
        QSTATE["paused"] = True
        return {"ok": True, "note": "Paused: the render on the GPU finishes, nothing new starts until resume."}
    if a == "resume":
        QSTATE["paused"] = False
        return {"ok": True}
    if a == "clear":  # waiting jobs only; never the one on the GPU
        n = 0
        for j in JOBS.values():
            if j["status"] == "queued" and not j.get("sent") and (priority is None or j.get("priority", "user") == priority):
                j["cancel"] = True
                n += 1
        return {"ok": True, "cancelled": n}
    j = JOBS.get(jid)
    if not j or j["status"] != "queued" or j.get("sent"):
        return {"error": "Only a job that is still waiting can be moved."}
    if a == "front":
        j["order"] = min([x.get("order", x["started"]) for x in JOBS.values()] + [time.time()]) - 1
        return {"ok": True}
    if a == "priority" and priority in PRIO:
        j["priority"] = priority
        return {"ok": True}
    return {"error": "Unknown queue action. Use pause, resume, clear, front or priority."}


def new_job(s, priority="user", target=None, **extra):
    jid = uuid.uuid4().hex[:8]
    try:  # kept with the job so the page can show the time left, whoever started it (page, chat agent, batch)
        minutes = minutes_for(s)[0]
    except Exception:
        minutes = None
    JOBS[jid] = dict({"status": "queued", "started": time.time(), "settings": s, "priority": priority,
                      "minutes": minutes}, **extra)
    threading.Thread(target=target or run_job, args=(jid, s), daemon=True).start()
    return jid


def wait_job(jid):
    while JOBS[jid]["status"] in ("queued", "rendering"):
        time.sleep(2)
    return JOBS[jid]


# ---------- self-check (review.py) and batches ----------
CHAINS, BATCHES = {}, {}


def settings_of(video):
    p = os.path.join(OUT, os.path.basename(video)[:-4] + ".json")
    if not os.path.exists(p):
        raise ValueError("No video called %s (or no saved settings for it)." % os.path.basename(video))
    s = json.load(open(p))
    for k in ("render_seconds", "gpu", "review"):
        s.pop(k, None)
    return s


def review_video(video, n=3):
    import review
    v = os.path.basename(video)
    s = settings_of(v)
    fr = review.frames(v, os.path.join(COMFY, "output"), n)
    strip = review.sheet(fr, os.path.join(review.FRAMES, v[:-4] + "_sheet.jpg"))
    return {"video": v, "frames": fr, "sheet": strip, "start_image": os.path.join(COMFY, "input", s["image"]) if s.get("image") else "",
            "prompt": s.get("prompt")}


def judge_video(video):
    import importlib, review
    importlib.reload(review)
    r = review_video(video)
    v = review.judge(r["start_image"], r["sheet"], r["prompt"])
    p = os.path.join(OUT, os.path.basename(video)[:-4] + ".json")
    h = json.load(open(p))
    h["review"] = v  # kept with the video: suggest_settings learns from it
    json.dump(h, open(p, "w"), indent=1)
    return dict(v, sheet=r["sheet"])


def fix_chain(cid, video, problems, tries, priority):
    """judge -> fix by failure type -> render again, max `tries` new renders; keeps every video, reports the best."""
    import review
    c = CHAINS[cid]
    try:
        cur, s = os.path.basename(video), settings_of(video)
        for i in range(tries + 1):
            c["now"] = "checking " + cur
            if i == 0 and problems:
                verdict = {"score": None, "ok": False, "problems": problems, "note": "problems given by the caller"}
            else:
                verdict = judge_video(cur)
            c["steps"].append(dict(video=cur, **{k: verdict.get(k) for k in ("score", "ok", "problems", "note")}))
            if verdict["ok"] or i == tries or c.get("cancel"):
                break
            s, why = review.fix(s, verdict["problems"])
            c["now"] = "rendering try %d: %s" % (i + 1, "; ".join(why))
            c["steps"][-1]["fix"] = why
            jid = new_job(dict(s), priority, chain=cid)
            c["job"] = jid
            j = wait_job(jid)
            if j["status"] != "done":
                c["steps"].append({"video": None, "error": j.get("error") or j["status"]})
                break
            cur, s = j["video"], settings_of(j["video"])
        scored = [x for x in c["steps"] if x.get("score") is not None]
        best = max(scored, key=lambda x: x["score"]) if scored else None
        c.update(status="done", best=best and best["video"], best_score=best and best["score"], now=None)
    except Exception as e:
        c.update(status="error", error=str(e)[:500], now=None)


def start_batch(items, priority="batch", label=""):
    import auto
    bid = "batch-" + uuid.uuid4().hex[:6]
    built = []
    for i, a in enumerate(items):  # check every item before starting any, so one typo doesn't leave half a batch
        try:
            s, _ = auto_render(dict(a))
        except (ValueError, TypeError) as e:
            raise ValueError("Item %d: %s" % (i + 1, e))
        if s.get("image") and not os.path.isfile(os.path.join(COMFY, "input", os.path.basename(s["image"]))):
            raise ValueError("Item %d: no picture called %r. Upload it first, or leave image empty to make it from the prompt alone." % (i + 1, s["image"]))
        built.append(dict(s, by="chat agent (batch)"))
    ids = [new_job(s, priority, batch=bid) for s in built]
    mins = sum(minutes_for(s)[0] for s in built)
    BATCHES[bid] = {"id": bid, "label": label, "jobs": ids, "minutes": round(mins), "started": time.time()}
    return BATCHES[bid]


def batch_view(bid):
    b = BATCHES.get(bid)
    if not b:
        return {"error": "No batch with id " + str(bid)}
    js = [dict(job_view(k, JOBS[k])) for k in b["jobs"]]
    st = [j["status"] for j in js]
    return dict(b, jobs=js, done=st.count("done"), failed=st.count("error"), cancelled=st.count("cancelled"),
                left=sum(s in ("queued", "rendering") for s in st), status="done" if all(s not in ("queued", "rendering") for s in st) else "running")


def suggest(model, goal):
    import auto
    g = (goal or "").lower()
    words = [("calm soft gentle subtle face portrait close-up", "Soft and calm"), ("still barely breathing blink photo", "Very soft"),
             ("energetic lively bouncy dance jump", "Lively"), ("big dynamic action fight run", "Strong"),
             ("same exact faithful keep look identical", "Stay close"), ("creative surprise wild", "Free and creative"),
             ("detail sharp eyes", "Sharper details"), ("slow motion slowmo", "Slow motion"), ("fast quick snappy", "Fast and snappy")]
    style = next((st for ws, st in words if any(w in g for w in ws.split())), "Balanced")
    past = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.json")), key=os.path.getmtime, reverse=True)[:200]:
        try:
            h = json.load(open(p))
        except Exception:
            continue
        r = h.get("review") or {}
        fam = {"small": "ltx", "ltx13": "ltx13", "vace": "vace"}.get(model, model)
        if r.get("score") is not None and (not model or h.get("family", "ltx") == fam or model in (h.get("unet"), h.get("checkpoint"))):
            past.append({"video": os.path.basename(p)[:-5] + ".mp4", "score": r["score"], "settings": h.get("easy"),
                         "movement": h.get("comp"), "seconds": round((int(h.get("frames", 49)) - 1) / 24), "minutes": round(h["render_seconds"] / 60, 1) if h.get("render_seconds") else None})
    past.sort(key=lambda x: -x["score"])
    return {"style": style, "quality": "Draft" if "test" in g or "try" in g else "Normal", "seconds": 2 if "short" in g else 3,
            "why": "Style picked from your goal's words; 2-3 s clips hold together best on small models.",
            "best_past_on_this_pc": past[:5] or "No checked videos yet (review_video/auto_fix record scores)."}


def _send_and_wait(job, graph):
    """Sends one graph to ComfyUI and waits for it. Returns its /history entry, or None if the job was cancelled."""
    pid = comfy("/prompt", {"prompt": graph, "client_id": CLIENT_ID})["prompt_id"]
    job.update(prompt_id=pid, status="queued")
    quiet_since = None
    while True:
        time.sleep(3)
        if job["status"] == "queued":  # ComfyUI runs one job at a time; show "rendering" only for the one on the GPU
            try:
                if any(q[1] == pid for q in comfy("/queue")["queue_running"]):
                    job.update(status="rendering", gpu_started=time.time())
            except Exception:
                pass
        if job.get("cancel"):
            if job["status"] == "rendering":
                comfy("/interrupt", {})
            else:  # still waiting inside ComfyUI behind someone else's job: take it out, don't stop theirs
                comfy("/queue", {"delete": [pid]})
            job.update(status="cancelled", ended=time.time())
            return None
        try:
            h = comfy("/history/" + pid)
            quiet_since = None
        except Exception as e:
            # A big decode (1080p "Best", or a PC short of memory) can keep ComfyUI from answering
            # for minutes while it still renders: wait. Fail only when it is really gone.
            quiet_since = quiet_since or time.time()
            gone = isinstance(getattr(e, "reason", e), ConnectionRefusedError)
            if time.time() - quiet_since > (90 if gone else 30 * 60):
                raise RuntimeError("ComfyUI stopped while rendering. Its log has the reason." if gone else
                                   "ComfyUI hasn't answered for 30 minutes. Restart it and try again.")
            job["note"] = "ComfyUI is busy (a big step like decoding the frames), still waiting"
            continue
        job.pop("note", None)
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                raise RuntimeError(json.dumps(st.get("messages"))[-1500:])
            return h[pid]


def _wait_turn(jid):
    job = JOBS[jid]
    while not my_turn(jid):  # our own queue: one of our jobs on ComfyUI at a time, by priority
        if job.get("cancel"):
            job.update(status="cancelled", ended=time.time())
            return False
        time.sleep(1)
    return True


LAST_MODEL = {"file": None}  # the video model the last render used


def _free_if_model_changed(s):
    """Chat agents render without the page's Model list, so the switch is caught here too: a different video
    model than last time clears the card first (ComfyUI applies /free before it runs the next job)."""
    model = s.get("unet") or s.get("checkpoint") or ""
    if LAST_MODEL["file"] and model and model != LAST_MODEL["file"]:
        try:
            comfy("/free", {"unload_models": True, "free_memory": True})
        except Exception:
            pass  # not worth failing a render over (ComfyUI answers /free with an empty body)
        time.sleep(1.5)  # /free wakes ComfyUI's idle worker; let it clear before our job arrives
    LAST_MODEL["file"] = model or LAST_MODEL["file"]


def run_job(jid, s):
    job = JOBS[jid]
    try:
        if int(s["seed"]) < 0:
            s["seed"] = random.randint(0, 2**31 - 1)
        job["settings"] = s
        if not _wait_turn(jid):
            return
        name = "ui_" + time.strftime("%m%d_%H%M%S") + "_s" + str(s["seed"]) + "_" + jid[:4]  # job id: same-second jobs never share a folder
        job["name"] = name
        _free_if_model_changed(s)
        if _send_and_wait(job, workflow(s, name)) is None:
            return
        os.makedirs(OUT, exist_ok=True)
        dst = os.path.join(OUT, name + ".mp4")
        to_mp4(os.path.join(COMFY, "output", name), int(s["fps"]), dst, 6000000 if not s.get("refine") else 12000000)
        took = round(time.time() - (job.get("gpu_started") or job["started"]))
        with open(os.path.join(OUT, name + ".json"), "w") as f:
            json.dump(dict(s, render_seconds=took, gpu=GPU_NAME()), f, indent=1)  # history for time estimates
        job.update(status="done", video=name + ".mp4", ended=time.time())
    except Exception as e:
        job.update(status="error", error=friendly_error(str(e)), ended=time.time())


# ---------- G: your own ComfyUI workflows (workflows.py) ----------
_OBJ = {}


def object_info():
    if not _OBJ or time.time() - _OBJ["at"] > 60:
        _OBJ.update(at=time.time(), info=comfy("/object_info"))
    return _OBJ["info"]


def list_workflows():
    import workflows
    workflows.seed_examples()
    out = []
    if os.path.isdir(workflows.WF_DIR):
        for f in sorted(os.listdir(workflows.WF_DIR)):
            if not f.endswith(".json"):
                continue
            item = {"name": f[:-5]}
            try:
                g = workflows.load(f[:-5])
                item.update(nodes=len(g), inputs=workflows.describe(g))
            except ValueError as e:
                item["error"] = str(e)
            out.append(item)
    return {"folder": workflows.WF_DIR, "workflows": out}


def check_workflow(name=None, graph=None):
    import workflows
    g = workflows.load(name) if name else workflows.as_graph(graph)
    out = workflows.check(g, object_info())
    out["inputs"] = workflows.describe(g)
    return out


def _input_file(value):
    """A local file path given for an image/video/audio input is copied into ComfyUI's input folder."""
    v = str(value)
    if os.path.isfile(v):
        name = os.path.basename(v)
        dst = os.path.join(COMFY, "input", name)
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(v):
            import shutil
            shutil.copy2(v, dst)
        return name
    return v


def start_workflow(name, inputs=None, priority="user", keep_seed=False, force=False):
    import workflows
    g = workflows.load(name)
    inputs = dict(inputs or {})
    for p in workflows.params(g):
        for k in list(inputs):
            if k.lower() == p["name"].lower() and p["name"].split("_")[0] in ("image", "video", "audio"):
                inputs[k] = _input_file(inputs[k])
    graph, used = workflows.apply(g, inputs, keep_seed)
    chk = workflows.check(graph, object_info())
    if not chk["ok"] and not force:
        return {"error": "This workflow would fail: " + "; ".join(
            chk["problems"][:5] + ["missing node type " + m["node"] for m in chk["missing_nodes"]][:5] +
            ["missing model file " + m["file"] for m in chk["missing_model_files"]][:5]) +
            ". Call check_workflow for details (or run with force=true to try anyway)."}
    s = {"workflow": workflows._safe(name), "inputs": used, "easy": "Workflow: " + workflows._safe(name)}
    jid = new_job(s, priority, target=run_workflow_job, graph=graph)
    return {"id": jid, "inputs": used, "minutes": minutes_for(s)[0], "starts_in_min": _starts_in(jid)}


def _starts_in(jid):
    for j in queue_view()["jobs"]:
        if j["id"] == jid:
            return j["starts_in_min"]
    return 0


def run_workflow_job(jid, s):
    import shutil
    import workflows
    job = JOBS[jid]
    try:
        if not _wait_turn(jid):
            return
        # your own workflow can load any model: each workflow counts as its own "model" for the switch check
        _free_if_model_changed({"unet": "workflow:" + str(s.get("workflow") or "")})
        entry = _send_and_wait(job, job.pop("graph"))
        if entry is None:
            return
        files = workflows.outputs(entry)
        if not files:
            raise RuntimeError("The workflow finished but saved no files (does it end in a Save node?).")
        took = round(time.time() - (job.get("gpu_started") or job["started"]))
        os.makedirs(OUT, exist_ok=True)
        base = "wf_" + time.strftime("%m%d_%H%M%S") + "_" + re.sub(r"[^\w\-]+", "-", s["workflow"])[:30] + "_" + jid[:4]
        copied = []
        for i, f in enumerate(files):
            ext = os.path.splitext(f)[1].lower()
            dst = os.path.join(OUT, base + ("_%d" % (i + 1) if len(files) > 1 else "") + ext)
            shutil.copy2(f, dst)
            copied.append(dst)
            if ext == ".mp4":
                with open(dst[:-4] + ".json", "w") as fh:
                    json.dump(dict(s, render_seconds=took, gpu=GPU_NAME(), source=f), fh, indent=1)
        vids = [c for c in copied if c.lower().endswith(".mp4")]
        job.update(status="done", files=copied, video=os.path.basename(vids[0]) if vids else None, ended=time.time(),
                   render_seconds=took)
        _remember_workflow_time(s["workflow"], took)
    except Exception as e:
        job.update(status="error", error=friendly_error(str(e)), ended=time.time())


WF_TIMES = os.path.join(DATA, "workflow-times.json")


def _remember_workflow_time(name, seconds):
    try:
        d = json.load(open(WF_TIMES))
    except Exception:
        d = {}
    d[name] = (d.get(name) or [])[-9:] + [seconds]
    with open(WF_TIMES, "w") as f:
        json.dump(d, f)


def save_workflow(name, graph, overwrite=False):
    import workflows
    return workflows.save(name, graph, overwrite)


def install_node(source, user_confirmed=False):
    import workflows
    if user_confirmed and busy():
        return {"error": "A video is rendering. Install the node pack when it's done (ComfyUI needs a restart after)."}
    return workflows.install(source, user_confirmed)


def gallery():
    items = []
    for p in sorted(glob.glob(os.path.join(OUT, "*.mp4")), key=os.path.getmtime, reverse=True):
        js = p[:-4] + ".json"
        s = json.load(open(js)) if os.path.exists(js) else {}
        items.append({"video": os.path.basename(p), "settings": s, "made": os.path.getmtime(p)})
    return items


# ---------- extra endpoints for the MCP server (mcp_server.py) ----------
COMFY_ROOT = paths.COMFY_ROOT  # ComfyUI_windows_portable for the portable build
COMFY_LOG = (os.environ.get("COMFYUI_LOG") or next((p for p in (os.path.join(os.path.dirname(COMFY_ROOT), "comfy.log"),) if os.path.exists(p)), None)
             or os.path.join(DATA, "comfy.log"))
DEFAULTS = os.path.join(DATA, "defaults.json")


def busy():
    """Jobs of this server, plus anything ComfyUI runs that someone else sent (another copy of this app, ComfyUI's own page)."""
    out = [k for k, j in JOBS.items() if j["status"] in ("queued", "rendering")]
    try:
        q = comfy("/queue")
        out += ["comfyui:" + x[1] for x in q.get("queue_running", []) + q.get("queue_pending", []) if x[1] not in {j.get("prompt_id") for j in JOBS.values()}]
    except Exception:
        pass
    return out


def comfy_up():
    try:
        comfy("/system_stats")
        return True
    except Exception:
        return False


def system_info():
    import shutil
    out = {"comfyui": comfy_up(), "comfyui_url": URL, "pipeline_busy": busy(), "free_disk_gb": round(shutil.disk_usage(COMFY).free / 1e9, 1)}
    if out["comfyui"]:
        st = comfy("/system_stats")
        out["gpus"] = [{"name": d.get("name"), "vram_gb": round(d.get("vram_total", 0) / 2**30, 1),
                        "vram_free_gb": round(d.get("vram_free", 0) / 2**30, 1)} for d in st.get("devices", [])]
        out["comfyui_version"] = st.get("system", {}).get("comfyui_version")
        q = comfy("/queue")
        out["comfy_queue"] = {"running": len(q.get("queue_running", [])), "waiting": len(q.get("queue_pending", []))}
    return out


def log_tail(n=80, grep=""):
    try:
        with open(COMFY_LOG, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 400000))
            lines = f.read().decode("utf-8", "replace").splitlines()
    except FileNotFoundError:
        return {"error": "No ComfyUI log at " + COMFY_LOG + " (set COMFYUI_LOG)."}
    if grep:
        lines = [l for l in lines if grep.lower() in l.lower()]
    return {"file": COMFY_LOG, "lines": lines[-max(1, min(int(n), 500)):]}


def defaults(update=None):
    d = {}
    if os.path.exists(DEFAULTS):
        d = json.load(open(DEFAULTS, encoding="utf-8"))
    if update is not None:
        d.update({k: v for k, v in update.items() if v is not None})
        for k in [k for k, v in update.items() if v is None]:
            d.pop(k, None)
        json.dump(d, open(DEFAULTS, "w", encoding="utf-8"), indent=1)
    return d


def diagnose(jid):
    j = JOBS.get(jid)
    if not j:
        return {"error": "No job with id " + str(jid) + " (jobs are kept until the pipeline restarts)."}
    out = {"id": jid, "status": j["status"], "error": j.get("error")}
    if j["status"] != "error":
        out["note"] = "This job did not fail."
        return out
    t, hints = (j.get("error") or "").lower(), []
    s = j.get("settings") or {}
    if "memory" in t:
        hints += ["Use quality 'Draft' or 'Normal' (smaller frames).", "Use fewer seconds.",
                  "Pick a smaller model (small, or a Q3/Q4 GGUF).", "Close other programs that use the graphics card, then call free_gpu_memory."]
    if "missing" in t or "not in list" in t:
        hints += ["Call list_models: the model or a helper file it needs is not installed. download_model fixes it."]
    if "comfyui stopped" in t or "connection" in t:
        hints += ["Call comfyui_status, then start_comfyui."]
    out["hints"] = hints or ["Read the ComfyUI log lines below for the cause."]
    out["log"] = log_tail(40, "")["lines"] if os.path.exists(COMFY_LOG) else []
    out["settings"] = {k: s.get(k) for k in ("family", "checkpoint", "unet", "width", "height", "frames", "steps")}
    return out


def start_comfyui(restart=False):
    if restart:
        if busy():
            return {"error": "A video is rendering. Wait for it or cancel it before restarting ComfyUI."}
        if comfy_up():
            pids = set()
            for line in subprocess.run(["netstat", "-ano", "-p", "tcp"], capture_output=True, text=True).stdout.splitlines():
                p = line.split()
                if len(p) >= 5 and p[1].endswith(":%d" % paths.PORT) and p[3] == "LISTENING":
                    pids.add(p[4])
            for pid in pids:
                subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
            for _ in range(20):
                if not comfy_up():
                    break
                time.sleep(0.5)
    elif comfy_up():
        return {"ok": True, "note": "ComfyUI is already running."}
    py = paths.comfy_python()
    if not py or not os.path.isfile(os.path.join(COMFY, "main.py")):
        return {"error": "Can't find ComfyUI at " + COMFY + ". Set COMFYUI_DIR (and COMFYUI_PYTHON if it has its own Python), "
                         "or start ComfyUI yourself."}
    portable = "python_embeded" in py
    log = open(COMFY_LOG, "ab")
    subprocess.Popen([py, "-s", os.path.join(COMFY, "main.py")] + (["--windows-standalone-build"] if portable else []) +
                     ["--listen", paths.HOST, "--port", str(paths.PORT), "--disable-auto-launch", "--preview-method", "none"],
                     cwd=COMFY_ROOT if portable else COMFY, stdout=log, stderr=subprocess.STDOUT,
                     creationflags=(0x08000000 | 0x00000200) if os.name == "nt" else 0)
    for _ in range(120):  # ComfyUI takes 20-60 s to load its nodes
        time.sleep(1)
        if comfy_up():
            return {"ok": True, "note": "ComfyUI started."}
    return {"error": "ComfyUI did not answer after 2 minutes. Read get_logs."}


def free_gpu():
    if busy():
        return {"error": "A video is rendering; freeing memory now would break it."}
    req = urllib.request.Request(URL + "/free", data=json.dumps({"unload_models": True, "free_memory": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()
    return {"ok": True, "note": "Models unloaded from the graphics card."}


def custom_by(v):
    for c in models_mgr.listed_models():
        if v in (c["file"], c["name"]):
            return c
    return None


def auto_render(a):
    """POST /api/render {"auto": {...}}: the page's Auto mode, for agents."""
    import importlib, auto
    importlib.reload(auto)  # presets.json edits apply without a restart
    a = {**defaults().get("render", {}), **{k: v for k, v in a.items() if v is not None}}
    model = a.pop("model", "small")
    cm = None if model in auto.MODELS else custom_by(model)
    if model not in auto.MODELS and not cm:
        raise ValueError("Unknown model %r. Use small, ltx13, vace, or an added model's name or file (list_models)." % model)
    return auto.easy_settings(model=model, custom=cm, **a), auto


def job_view(k, j):
    now = time.time()
    left = None  # seconds, from this PC's usual time for these settings; can go below 0 on a slow run
    if j.get("status") == "rendering" and j.get("minutes") and j.get("gpu_started"):
        left = round(j["minutes"] * 60 - (now - j["gpu_started"]))
    return dict({x: y for x, y in j.items() if x != "cancel"}, id=k, progress=PROGRESS.get(j.get("prompt_id")),
                elapsed=round((j.get("ended") or now) - (j.get("gpu_started") or j["started"])), left_s=left)


# ---------- phase 3: editing (edit.py), run as edit jobs ----------
EDITS = {}
EDIT_LOCK = threading.Lock()  # one ffmpeg edit at a time: they are CPU-heavy and build_video tracks its files per thread


def run_edit(op, args):
    """Starts an edit; returns its id. Quick edits finish within the HTTP wait below; long ones (voice, extend,
    build_video) are followed with GET /api/edit?id=."""
    fn = _edit_ops().get(op)
    if not fn:
        raise ValueError("Unknown edit %r." % op)
    eid = "edit-" + uuid.uuid4().hex[:6]
    EDITS[eid] = {"id": eid, "op": op, "status": "running", "started": time.time()}

    def go():
        try:
            with EDIT_LOCK if op not in ("extend_clip", "upscale_ltx") else _nolock():
                r = fn(**args)
            EDITS[eid].update(status="done", result=r, ended=time.time())
        except Exception as e:
            import edit as E
            msg = str(e) if isinstance(e, (E.EditError, ValueError)) else "%s: %s" % (type(e).__name__, str(e)[:400])
            EDITS[eid].update(status="error", error=msg, ended=time.time())
    threading.Thread(target=go, daemon=True).start()
    return eid


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _render_from(settings, priority="user"):
    jid = new_job(settings, priority)
    j = wait_job(jid)
    if j["status"] != "done":
        raise ValueError("The render failed: " + (j.get("error") or j["status"]))
    return j["video"]


def extend_clip(video, prompt, seconds=None, transition="cut"):
    """GPU: the clip's last frame becomes the start picture of a new clip with `prompt`, then both are joined."""
    import edit
    s = settings_of(video)
    if s.get("edit"):
        raise ValueError("extend_clip needs a video made by the pipeline (it reuses its model and settings), not an edit.")
    png = "ext_%s.png" % os.path.basename(video)[:-4]
    edit.last_frame(video, os.path.join(COMFY, "input", png))
    s.update(image=png, prompt=prompt, seed=-1)
    s["easy"] = "Extend of " + os.path.basename(video)
    if seconds:
        fps = 16 if s.get("family") == "vace" else 24
        s["frames"] = int(seconds) * fps + 1
    new = _render_from(s)
    info = edit.probe(os.path.join(OUT, new))
    tail = edit.trim(new, 1.0 / info["fps"], None)["video"]  # its first frame repeats the old last frame
    r = edit.join([video, tail], transition, 0.3)
    tmp = os.path.join(OUT, "tmp")
    os.makedirs(tmp, exist_ok=True)
    for ext in (".mp4", ".json"):  # the trimmed piece is only a step; keep it out of the gallery
        p = os.path.join(OUT, tail[:-4] + ext)
        if os.path.exists(p):
            os.replace(p, os.path.join(tmp, os.path.basename(p)))
    return dict(r, new_clip=new)


def upscale_ltx(video):
    """GPU: renders the same clip again (same seed) with the LTX second pass to 1080p. Slow (30-50 min on 4 GB)."""
    s = settings_of(video)
    if s.get("edit") or s.get("family") not in (None, "ltx", "ltx13"):
        raise ValueError("The LTX upscale works for LTX videos made by the pipeline. Use method=\"simple\" for others.")
    s.update(refine=True)
    s["easy"] = (s.get("easy") or "") + " · upscaled"
    v = _render_from(s)
    return {"video": v, "path": os.path.join(OUT, v)}


def lipsync(video, audio):
    vram = models_mgr.vram_gb() or 0
    raise ValueError("Lip-sync (LatentSync) needs a graphics card with 8 GB or more and a real human face; this PC has %.0f GB. "
                     "For cartoon characters, add_voice and a closed-mouth prompt work best." % vram)


def build_video(clips, transition="crossfade", shape=None, voice=None, captions=None, music=None, title=None,
                end_card=None, export=None):
    """One call for a finished video. Order: clips (trim/speed) -> join -> shape -> voice -> captions -> cards -> music -> export.
    Only the final file stays in the gallery; the in-between files go to out/tmp."""
    import edit
    edit.TRACK.files, steps = [], []
    try:
        parts = []
        for c in clips:
            c = {"video": c} if isinstance(c, str) else dict(c)
            v = c["video"]
            if c.get("start") or c.get("end"):
                v = edit.trim(v, c.get("start") or 0, c.get("end"))["video"]
            if c.get("speed") and float(c["speed"]) != 1:
                v = edit.speed(v, c["speed"])["video"]
            parts.append(v)
        cur = edit.join(parts, transition, 0.4)["video"] if len(parts) > 1 else parts[0]
        steps.append("joined %d clips (%s)" % (len(parts), transition))
        if shape:
            cur = edit.reformat(cur, shape, "blur")["video"]
            steps.append("shape " + shape)
        if voice:
            vo = voice if isinstance(voice, dict) else {"text": voice}
            aud = vo.get("audio") or edit.make_voice(vo["text"], vo.get("voice", "andrew"), vo.get("engine", "edge"), vo.get("sample"),
                                                     vo.get("language", "en"))["path"]
            cur = edit.add_voice(cur, aud, vo.get("volume", 1.0), vo.get("start", 0.3))["video"]
            steps.append("voice")
        if captions:
            cur = edit.add_captions(cur, captions if captions != True else "auto", "bold")["video"]
            steps.append("captions")
        if title:
            cur = edit.title_card(cur, title, None, 1.8)["video"]
            steps.append("title card")
        if end_card:
            cur = edit.end_card(cur, end_card if isinstance(end_card, str) else "Follow for more", None, 2.2)["video"]
            steps.append("end card")
        if music:
            mu = music if isinstance(music, dict) else {"file": music}
            cur = edit.add_music(cur, mu["file"], mu.get("volume", 0.25), 1.0, True)["video"]
            steps.append("music")
        if export:
            cur = edit.export(cur, export)["video"]
            steps.append("export for " + export)
        final = os.path.join(OUT, cur)
    finally:
        made, edit.TRACK.files = edit.TRACK.files, None
    keep = os.path.join(OUT, "tmp", "build_" + time.strftime("%m%d_%H%M%S"))
    for p in made:
        if os.path.abspath(p) != os.path.abspath(final):
            os.makedirs(keep, exist_ok=True)
            for ext in (".mp4", ".json"):
                if os.path.exists(p[:-4] + ext):
                    os.replace(p[:-4] + ext, os.path.join(keep, os.path.basename(p)[:-4] + ext))
    info = edit.probe(final)
    return {"video": cur, "path": final, "seconds": round(info["duration"], 2), "width": info.get("width"), "height": info.get("height"),
            "mb": round(os.path.getsize(final) / 1e6, 2), "steps": steps, "in_between_files": keep}


def _edit_ops():
    import edit
    return {"trim": edit.trim, "speed": edit.speed, "reverse": edit.reverse, "loop": edit.loop, "boomerang": edit.boomerang,
            "fade": edit.fade, "color_adjust": edit.color_adjust, "crop_zoom": edit.crop_zoom, "smooth_motion": edit.smooth_motion,
            "upscale_simple": edit.upscale_simple, "join": edit.join, "reformat": edit.reformat, "export": edit.export,
            "thumbnail": edit.thumbnail, "make_gif": edit.make_gif, "extract_audio": edit.extract_audio, "remove_audio": edit.remove_audio,
            "add_music": edit.add_music, "add_voice": edit.add_voice, "make_voice": edit.make_voice, "add_text": edit.add_text,
            "add_logo": edit.add_logo, "add_captions": edit.add_captions, "title_card": edit.title_card, "end_card": edit.end_card,
            "card": edit.card, "build_video": build_video, "extend_clip": extend_clip, "upscale_ltx": upscale_ltx, "lipsync": lipsync}



def trash_video(name):
    name = os.path.basename(name)
    src = os.path.join(OUT, name)
    if not name.endswith(".mp4") or not os.path.exists(src):
        return {"error": "No video called " + name}
    t = os.path.join(OUT, "trash")
    os.makedirs(t, exist_ok=True)
    for ext in (".mp4", ".json"):
        p = src[:-4] + ext
        if os.path.exists(p):
            os.replace(p, os.path.join(t, os.path.basename(p)))
    return {"ok": True, "note": "Moved to " + t + " (not deleted; empty that folder to free the space)."}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if ctype.startswith(("text/html", "application/json")):  # always fresh: a cached page hid new renders and fixes
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?")[0]
        try:
            if p == "/":
                page = open(os.path.join(HERE, "index.html"), "rb").read()
                presets = open(os.path.join(HERE, "presets.json"), "rb").read()
                return self.send(200, page.replace(b"/*@PRESETS*/null", presets), "text/html; charset=utf-8")
            q = dict(urllib.parse.parse_qsl(self.path.partition("?")[2]))
            if p == "/api/system":
                return self.send(200, system_info())
            if p == "/api/workflows":
                return self.send(200, list_workflows())
            if p == "/api/workflow/check":
                return self.send(200, check_workflow(q.get("name")))
            if p == "/api/logs":
                return self.send(200, log_tail(q.get("n", 80), q.get("grep", "")))
            if p == "/api/defaults":
                return self.send(200, defaults())
            if p == "/api/diagnose":
                return self.send(200, diagnose(q.get("id")))
            if p == "/api/job":
                j = JOBS.get(q.get("id"))
                return self.send(200, job_view(q["id"], j) if j else {"error": "No job with id " + str(q.get("id"))})
            if p == "/api/edit":
                e = EDITS.get(q.get("id"))
                return self.send(200, e or {"error": "No edit with id " + str(q.get("id"))})
            if p == "/api/queue":
                return self.send(200, queue_view())
            if p == "/api/batch":
                return self.send(200, batch_view(q.get("id")))
            if p == "/api/fix":
                c = CHAINS.get(q.get("id"))
                return self.send(200, c or {"error": "No fix chain with id " + str(q.get("id"))})
            if p == "/api/review":
                try:
                    return self.send(200, review_video(q.get("video") or "", int(q.get("n", 3))))
                except Exception as e:
                    return self.send(200, {"error": str(e)[:300]})
            if p == "/api/suggest":
                return self.send(200, suggest(q.get("model") or "", q.get("goal") or ""))
            if p == "/api/presets":
                import auto
                return self.send(200, {"models": auto.MODELS, "styles": [{"name": s["name"], "help": s["help"]} for s in auto.STYLES],
                                       "qualities": [{"name": x["name"], "help": x["help"]} for x in auto.QUALITY],
                                       "cameras": list(auto.CAM), "speeds": auto.SPEEDS, "seconds": auto.LENGTHS, "avoid": auto.AVOID})
            if p == "/api/options":
                return self.send(200, options())
            if p == "/api/models":
                return self.send(200, models_mgr.status())
            if p == "/api/jobs":
                return self.send(200, [job_view(k, j) for k, j in sorted(JOBS.items(), key=lambda x: -x[1]["started"])])
            if p == "/api/gallery":
                return self.send(200, gallery())
            if p.startswith("/video/"):
                # Byte ranges (206): Chrome's player asks for pieces; a plain 200 made it re-fetch
                # the newest video in a loop forever (spinner every few seconds, David 11-18).
                f = os.path.join(OUT, os.path.basename(p))
                data = open(f, "rb").read()
                rng = self.headers.get("Range", "")
                if rng.startswith("bytes="):
                    a, _, b = rng[6:].split(",")[0].partition("-")
                    start = int(a) if a else max(0, len(data) - int(b))
                    end = min(int(b), len(data) - 1) if (a and b) else len(data) - 1
                    chunk = data[start:end + 1]
                    self.send_response(206)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, len(data)))
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("Cache-Control", "max-age=3600")
                    self.end_headers()
                    return self.wfile.write(chunk)
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                return self.wfile.write(data)
            if p.startswith("/image/"):
                f = os.path.join(COMFY, "input", os.path.basename(urllib.request.unquote(p[7:])))
                return self.send(200, open(f, "rb").read(), "image/png")
            if p == "/api/ping":
                # The MCP's "is the pipeline up?" check. Never touches ComfyUI: while ComfyUI loads a
                # model it answers slowly, and that once made the agent say the pipeline wasn't running.
                return self.send(200, {"ok": True})
            if p == "/api/comfy":
                try:
                    comfy("/system_stats")
                    return self.send(200, {"up": True})
                except Exception:
                    return self.send(200, {"up": False})
        except FileNotFoundError:
            return self.send(404, {"error": "not found"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path.startswith("/api/workflow/"):
            try:
                op = self.path.rsplit("/", 1)[1]
                if op == "run":
                    prio = body.get("priority") or "user"
                    if prio not in PRIO:
                        return self.send(200, {"error": "priority must be user or batch"})
                    return self.send(200, start_workflow(body.get("name"), body.get("inputs"), prio,
                                                         bool(body.get("keep_seed")), bool(body.get("force"))))
                if op == "check":
                    return self.send(200, check_workflow(body.get("name"), body.get("workflow")))
                if op == "save":
                    return self.send(200, save_workflow(body.get("name"), body.get("workflow"), bool(body.get("overwrite"))))
                if op == "install":
                    return self.send(200, install_node(body.get("source"), bool(body.get("user_confirmed"))))
                return self.send(200, {"error": "Unknown workflow action " + op})
            except ValueError as e:
                return self.send(200, {"error": str(e)})
            except Exception as e:
                return self.send(200, {"error": friendly_error(str(e))})
        if self.path in ("/api/render", "/api/estimate"):
            # {"auto": {...}} = Auto mode (agents); anything else = full settings (the page, rerender).
            # "priority": "user" (default) or "batch"; batch jobs wait while user jobs go first.
            prio = body.pop("_priority", None) or body.pop("priority", None) or "user"
            if prio not in PRIO:
                return self.send(200, {"error": "priority must be user or batch"})
            # Who started it, shown on the render and kept in its gallery card: Auto mode by API = an agent.
            by = "chat agent" if "auto" in body else "you, on this page"
            if "auto" in body:
                try:
                    body, _ = auto_render(body["auto"])
                except (ValueError, TypeError) as e:
                    return self.send(200, {"error": str(e)})
                if self.path == "/api/render" and body.get("image") and not os.path.isfile(os.path.join(COMFY, "input", os.path.basename(body["image"]))):
                    return self.send(200, {"error": "No picture called %r. Upload it first (upload_image), or leave image empty to make the video from the prompt alone." % body["image"]})
            body["by"] = by
            minutes, based_on = minutes_for(body)
            wait = queue_view()["all_done_in_min"]
            if self.path == "/api/estimate":
                return self.send(200, {"minutes": minutes, "based_on_renders": based_on, "queue_wait_min": wait, "settings": body.get("easy")})
            jid = new_job(body, prio)
            return self.send(200, {"id": jid, "minutes": minutes, "starts_in_min": round(wait), "settings": body})
        if self.path == "/api/edit":  # {"op": ..., "args": {...}}; waits up to 40 s, else returns the running edit's id
            try:
                eid = run_edit(body.get("op"), body.get("args") or {})
            except ValueError as e:
                return self.send(200, {"error": str(e)})
            end = time.time() + min(40, float(body.get("wait", 40)))
            while EDITS[eid]["status"] == "running" and time.time() < end:
                time.sleep(0.3)
            return self.send(200, EDITS[eid])
        if self.path == "/api/queue":
            return self.send(200, queue_action(body.get("action"), body.get("id"), body.get("priority")))
        if self.path == "/api/batch":
            try:
                if body.get("estimate_only"):
                    import auto
                    mins = [minutes_for(auto_render(dict(a))[0])[0] for a in body.get("items") or []]
                    return self.send(200, {"clips": len(mins), "minutes": round(sum(mins)), "queue_wait_min": queue_view()["all_done_in_min"]})
                return self.send(200, start_batch(body.get("items") or [], body.get("priority") or "batch", body.get("label") or ""))
            except (ValueError, TypeError) as e:
                return self.send(200, {"error": str(e)})
        if self.path == "/api/judge":
            try:
                return self.send(200, judge_video(body.get("video") or ""))
            except Exception as e:
                return self.send(200, {"error": str(e)[:500]})
        if self.path == "/api/fix":
            v = os.path.basename(body.get("video") or "")
            try:
                settings_of(v)
            except ValueError as e:
                return self.send(200, {"error": str(e)})
            cid = "fix-" + uuid.uuid4().hex[:6]
            CHAINS[cid] = {"id": cid, "video": v, "status": "running", "steps": [], "started": time.time()}
            tries = max(1, min(int(body.get("tries") or 2), 3))
            threading.Thread(target=fix_chain, args=(cid, v, body.get("problems") or [], tries, body.get("priority") or "user"), daemon=True).start()
            return self.send(200, CHAINS[cid])
        if self.path == "/api/defaults":
            return self.send(200, defaults(body))
        if self.path == "/api/free":
            try:
                return self.send(200, free_gpu())
            except Exception as e:
                return self.send(200, {"error": "ComfyUI didn't answer: " + str(e)[:200]})
        if self.path == "/api/comfy/start":
            return self.send(200, start_comfyui(bool(body.get("restart"))))
        if self.path == "/api/delete":
            return self.send(200, trash_video(body.get("video") or ""))
        if self.path == "/api/models":
            a = body.get("action")
            try:
                if a == "check":
                    return self.send(200, models_mgr.check(body.get("file") or ""))
                if a == "pickadd":  # "+" button: Windows file window, then add the file where it is
                    p = models_mgr.pick()
                    return self.send(200, models_mgr.add_path(p) if p else {"cancel": True})
                if a == "add":
                    return self.send(200, models_mgr.add(body.get("name"), body.get("file") or ""))
                if a == "remove":
                    return self.send(200, models_mgr.remove(body.get("file") or ""))
                if a == "addpath":  # an agent adds a model file by its full path (the same as the "+" button)
                    return self.send(200, models_mgr.add_path(body.get("path") or "", body.get("name")))
                if a == "hfcheck":
                    return self.send(200, models_mgr.hf_check(body.get("repo") or "", body.get("file") or ""))
                if a == "hfsearch":
                    return self.send(200, models_mgr.hf_search(body.get("q") or ""))
                if a == "hfdownload":
                    return self.send(200, models_mgr.hf_download(body.get("repo") or "", body.get("file") or ""))
                if a == "download":
                    return self.send(200, models_mgr.start_download(body.get("id")))
                if a == "cancel":
                    models_mgr.DL["cancel"] = True
                    return self.send(200, {"ok": True})
                if a == "open":
                    return self.send(200, models_mgr.open_folder(body.get("folder")))
            except Exception as e:
                return self.send(200, {"error": "Something went wrong: " + str(e)[:300]})
            return self.send(200, {"error": "unknown action"})
        if self.path == "/api/cancel":
            if body.get("id") in JOBS:
                JOBS[body["id"]]["cancel"] = True
            return self.send(200, {"ok": True})
        if self.path == "/api/genprompt":
            img = os.path.join(COMFY, "input", os.path.basename(body.get("image") or ""))
            if not body.get("image") or not os.path.isfile(img):
                return self.send(200, {"error": "The prompt writer looks at your picture. Pick or upload one, or write the prompt yourself for a video without a picture."})
            if not (body.get("idea") or "").strip():
                return self.send(200, {"error": "Write your idea in the prompt box first."})
            try:
                import importlib, promptgen
                importlib.reload(promptgen)  # edits to the prompt rules apply without a server restart
                return self.send(200, promptgen.generate(img, body["idea"], body.get("kind") or "prompt", body.get("negative") or ""))
            except Exception as e:
                return self.send(200, {"error": str(e)[:400]})
        if self.path == "/api/upload":
            name = os.path.basename(body["name"])
            with open(os.path.join(COMFY, "input", name), "wb") as f:
                f.write(base64.b64decode(body["data"].split(",")[-1]))
            return self.send(200, {"name": name})
        self.send(404, {"error": "not found"})


def main():
    print(f"AI Video Pipeline UI on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()


if __name__ == "__main__":
    main()
