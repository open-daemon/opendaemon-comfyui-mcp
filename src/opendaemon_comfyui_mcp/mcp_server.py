"""OpenDaemon ComfyUI MCP - lets any AI agent make videos on your own PC with ComfyUI.

A thin front over the video pipeline server (server.py, http://127.0.0.1:8190), so the web page and
every agent share one queue. Talks to ComfyUI only through its HTTP API.

Run (stdio):  python mcp_server.py        needs Python 3.10+ and `pip install mcp`
Settings (environment):
  PIPELINE_URL   default http://127.0.0.1:8190
  PIPELINE_AUTOSTART=0  never start the pipeline server itself (an app that runs it sets this)
  PIPELINE_PYTHON  the Python that runs server.py if it is not running yet (default: "python" on PATH, else this one)
MIT licence.
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# Python packages shipped next to us (OpenDaemon ships mcp + its helpers). Embedded Pythons ignore
# PYTHONPATH, so the folder is added here, before the first import that needs it.
if os.environ.get("COMFY_MCP_PACKAGES"):
    import site
    sys.path.insert(0, os.environ["COMFY_MCP_PACKAGES"])
    site.addsitedir(os.environ["COMFY_MCP_PACKAGES"])  # also runs their .pth files (pywin32 needs its own)

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import ToolAnnotations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths  # noqa: E402  where videos are kept (COMFY_MCP_DATA)
PIPE = os.environ.get("PIPELINE_URL", "http://127.0.0.1:8190").rstrip("/")
VERSION = "0.1.0"
READ = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
WRITE = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)
DESTROY = ToolAnnotations(read_only_hint=False, destructive_hint=True, open_world_hint=False)
WEB = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=True)

GUIDE = """How to make a video with this server
1. comfyui_status - is ComfyUI running, which graphics card, how much memory. start_comfyui if it is down.
2. list_models - what is installed. On a 4 GB card start with model "small".
3. upload_image (optional) - a picture to start from (image to video). Use the name it returns.
   No picture? render with the prompt alone (text to video): describe the whole scene then.
4. write_prompt (optional, needs a picture) - turns a short idea into a good video prompt, looking at the picture.
5. render - picks every setting from a style and a quality. Returns a job id and a rough time.
6. job_status - waits up to 50 s per call and reports progress. Call it again until status is done.
   The video's path is in the answer. One video renders at a time; others wait in the queue.
If a job fails: diagnose_job. Out of memory: quality "Draft", fewer seconds, or a smaller model.

Prompt tips (LTX-Video): describe ONE simple action in the present tense, then the look and the camera.
Say what moves and how ("slowly raises one hand", "the antennae wobble softly"). Keep it under 120 words.
Long clips (5 s+) drift at the end: make short clips instead. The camera choice is added for you.
Without a picture the model invents everything, so also say who or what is there, what it looks like, the place,
the light and the style ("a fluffy white kitten on a grey rug in a sunny living room, soft daylight, realistic").
A picture gives steadier, sharper clips; ask the user for one when the look matters."""

mcp = MCPServer("OpenDaemon ComfyUI MCP", version=VERSION, instructions=GUIDE,
                website_url="https://open-daemon.com")


# ---------- talking to the pipeline server ----------
class PipelineError(ToolError):
    """A plain-language problem the agent should read (shown as the tool's error text)."""


def _call(path, body=None, timeout=60):
    req = urllib.request.Request(PIPE + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _pipeline_up():
    """True when the pipeline server answers. Asks the pipeline only, not ComfyUI: ComfyUI can be slow
    for a while after a render starts (loading the model), and a slow ComfyUI is not a stopped pipeline.
    Tries twice so one slow moment doesn't count as down."""
    for attempt in range(2):
        try:
            _call("/api/ping", timeout=8)
            return True
        except urllib.error.HTTPError:
            return True  # it answered (an older pipeline without /api/ping)
        except Exception:
            if attempt == 0:
                time.sleep(1.5)
    return False


def _ensure_pipeline():
    if _pipeline_up():
        return
    # PIPELINE_AUTOSTART=0: an app owns the pipeline (starts it, and stops it when it closes), so never
    # leave a copy running that nobody will stop. PIPELINE_DOWN_HINT tells the model what the user should do.
    if os.environ.get("PIPELINE_AUTOSTART", "1").strip() == "0":
        raise PipelineError(os.environ.get("PIPELINE_DOWN_HINT") or "The video pipeline server isn't running at " + PIPE + ".")
    # installed as a package: the same Python; the dev copy: "python" on PATH (has edge-tts there)
    packaged = paths.DATA != HERE
    py = os.environ.get("PIPELINE_PYTHON") or (sys.executable if packaged else "python")
    # A bare name like "python" must be looked up on PATH here: Windows would first look next to THIS Python
    # (ComfyUI's embedded one, which can't import modules from the script's folder).
    py = shutil.which(py) or py if not os.path.dirname(py) else py
    subprocess.Popen([py, os.path.join(HERE, "server.py")], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=(0x08000000 | 0x00000200) if os.name == "nt" else 0)
    for _ in range(40):
        time.sleep(0.5)
        if _pipeline_up():
            return
    raise PipelineError("The video pipeline server did not start. Run `python server.py` in " + HERE + " and try again.")


def api(path, body=None, timeout=60):
    """Calls the pipeline; turns its {"error": ...} answers into a tool error the agent can read."""
    _ensure_pipeline()
    try:
        r = _call(path, body, timeout)
    except Exception as e:
        raise PipelineError("The video pipeline server didn't answer %s (%s)." % (path.split("?")[0], str(e)[:200]))
    if isinstance(r, dict) and r.get("error"):
        raise PipelineError(r["error"])
    return r


def models_action(action, **kw):
    return api("/api/models", {"action": action, **kw}, timeout=120)


def video_path(name):
    return os.path.join(paths.OUT, os.path.basename(name))


def _short_job(j):
    out = {"id": j["id"], "status": j["status"], "elapsed_s": j.get("elapsed")}
    p = j.get("progress")
    if p and j["status"] == "rendering":
        out["progress"] = "%d/%d steps" % (p["value"], p["max"]) + (" (pass %d)" % p["pass_no"] if p.get("pass_no", 1) > 1 else "")
    if j.get("video"):
        out.update(video=j["video"], path=video_path(j["video"]), url=PIPE + "/video/" + j["video"])
    if j.get("files") and (len(j["files"]) > 1 or not j.get("video")):
        out["files"] = j["files"]
    if j.get("error"):
        out["error"] = j["error"]
    if j.get("note"):
        out["note"] = j["note"]
    s = j.get("settings") or {}
    if s.get("easy"):
        out["settings"] = s["easy"]
    return out


# ---------- A. system and ComfyUI ----------
@mcp.tool(annotations=READ)
def comfyui_status() -> dict:
    """Is ComfyUI running, which graphics card and how much memory it has, free disk space, and what is rendering."""
    return api("/api/system")


@mcp.tool(annotations=WRITE)
def start_comfyui() -> dict:
    """Starts ComfyUI if it is not running (takes 20-60 s). Does nothing if it already runs."""
    return api("/api/comfy/start", {}, timeout=150)


@mcp.tool(annotations=DESTROY)
def restart_comfyui(user_confirmed: bool = False) -> dict:
    """Stops and starts ComfyUI again. Fixes a stuck ComfyUI. Refused while a video renders.
    Ask the user first, then call with user_confirmed=true."""
    if not user_confirmed:
        return {"needs_confirmation": "This stops ComfyUI for everyone using it on this PC (about 1 minute). Ask the user, then call again with user_confirmed=true."}
    return api("/api/comfy/start", {"restart": True}, timeout=180)


@mcp.tool(annotations=WRITE)
def free_gpu_memory() -> dict:
    """Unloads models from the graphics card. Use after an out-of-memory error or before another program needs the GPU.
    Refused while a video renders."""
    return api("/api/free", {})


@mcp.tool(annotations=READ)
def get_logs(lines: int = 80, contains: str = "") -> dict:
    """The last lines of the ComfyUI log (max 500). contains= keeps only lines with that text, e.g. "error"."""
    return api("/api/logs?" + urllib.parse.urlencode({"n": lines, "grep": contains}))


@mcp.tool(annotations=READ)
def get_defaults() -> dict:
    """Your saved defaults for render (e.g. model, quality, style). They apply when render leaves a setting out."""
    return api("/api/defaults")


@mcp.tool(annotations=WRITE)
def set_defaults(model: str | None = None, style: str | None = None, quality: str | None = None, camera: str | None = None,
                 seconds: int | None = None, avoid: str | None = None, clear: bool = False) -> dict:
    """Saves render defaults so later renders can leave them out. clear=true forgets all of them."""
    cur = api("/api/defaults").get("render", {})
    if clear:
        cur = {}
    cur.update({k: v for k, v in dict(model=model, style=style, quality=quality, camera=camera, seconds=seconds, avoid=avoid).items() if v is not None})
    return api("/api/defaults", {"render": cur})


# ---------- B. models ----------
@mcp.tool(annotations=READ)
def list_models() -> dict:
    """What models are installed and what can be downloaded, with size, the card they need, and if we tested them.
    "best" marks the best one for this PC's graphics card. Use a model's "use_as" value in render(model=...)."""
    s = api("/api/models")
    cat = []
    for m in s["catalog"]:
        if m.get("type") is None and m["id"] != "ltx-upscaler":
            continue
        cat.append({"id": m["id"], "name": m["name"], "installed": m["installed"], "download_gb": m["need_gb"],
                    "needs_card_gb": m["card"], "tested": m["tested"], "best_for_this_pc": m["best"],
                    "use_as": m["builtin"] or (m["main"] if m["installed"] else None), "about": m.get("help")})
    return {"graphics_card_gb": s["vram"], "free_disk_gb": s["free"], "catalog": cat,
            "added_by_user": [{"name": c["name"], "file": c["file"], "kind": c.get("kind"), "use_as": c["file"]} for c in s["custom"]],
            "download_running": s["download"] and {k: s["download"].get(k) for k in ("id", "name", "done", "total", "error", "finished")}}


@mcp.tool(annotations=WEB)
def download_model(model_id: str, user_confirmed: bool = False) -> dict:
    """Downloads a model from our list (ids from list_models), with any helper files it needs.
    Big downloads: tell the user the size and call again with user_confirmed=true."""
    m = next((x for x in api("/api/models")["catalog"] if x["id"] == model_id), None)
    if not m:
        raise PipelineError("No model %r in the list. Call list_models for the ids." % model_id)
    if m["installed"]:
        return {"ok": True, "note": m["name"] + " is already installed."}
    if not user_confirmed:
        return {"needs_confirmation": "%s needs %.1f GB of downloads. Ask the user, then call again with user_confirmed=true." % (m["name"], m["need_gb"])}
    return models_action("download", id=model_id)


@mcp.tool(annotations=READ)
def download_status() -> dict:
    """Progress of the running model download (one at a time)."""
    d = api("/api/models")["download"]
    if not d:
        return {"running": False}
    pct = round(100 * d["done"] / d["total"], 1) if d.get("total") else None
    return {"running": not d.get("finished") and not d.get("error"), "name": d.get("name"), "percent": pct,
            "done_gb": round(d["done"] / 1e9, 2), "total_gb": round((d.get("total") or 0) / 1e9, 2),
            "finished": d.get("finished"), "error": d.get("error")}


@mcp.tool(annotations=WRITE)
def cancel_download() -> dict:
    """Stops the running model download. What was downloaded is kept, so starting it again resumes."""
    return models_action("cancel")


@mcp.tool(annotations=WEB)
def search_models(query: str = "") -> dict:
    """Searches Hugging Face for video models that fit this PC's graphics card (bigger ones are hidden).
    These are NOT tested by us; check_model reads a file's header first. Empty query = LTX-Video and Wan VACE models."""
    r = models_action("hfsearch", q=query)
    repos = [{"repo": x["repo"], "downloads": x["downloads"], "likes": x["likes"],
              "files": [{"file": f["file"], "gb": f["gb"], "in_our_list": f.get("ours")} for f in x["files"]]} for x in r["repos"]]
    return {"card_gb": r["vram"], "max_file_gb": r["cap"], "hidden_too_big": r["hidden"], "repos": repos,
            "warning": "Untested models can fail or look bad. Prefer models from list_models."}


@mcp.tool(annotations=WEB)
def check_model(file: str = "", repo: str = "") -> dict:
    """Says if a model file can run here and why not if it can't, without loading it.
    A file on this PC: file="C:/path/model.gguf" or a file name in ComfyUI's models folder.
    A file on Hugging Face (from search_models): repo="owner/name" and file="path/in/repo.gguf"."""
    if repo:
        return models_action("hfcheck", repo=repo, file=file)
    return models_action("check", file=file)


@mcp.tool(annotations=WRITE)
def add_model_file(path: str, name: str = "") -> dict:
    """Adds a model file that is already on this PC, where it is (a link, no copy). It is checked first.
    After this, render(model=<the file name>) uses it."""
    return models_action("addpath", path=path, name=name or None)


@mcp.tool(annotations=DESTROY)
def remove_model(file: str, user_confirmed: bool = False) -> dict:
    """Removes a model the user added (from list_models "added_by_user"). Only our link is removed, never the user's file.
    Ask the user first, then call with user_confirmed=true."""
    if not user_confirmed:
        return {"needs_confirmation": "This removes %s from the app. Ask the user, then call again with user_confirmed=true." % file}
    return models_action("remove", file=file)


# ---------- C. pictures and prompts ----------
@mcp.tool(annotations=WEB)
def upload_image(path: str = "", url: str = "", base64_data: str = "", name: str = "") -> dict:
    """Puts a picture into ComfyUI's input folder. Give ONE of: path (a file on this PC), url (http/https),
    base64_data (the image bytes; give a name like "cat.png"). Returns the name to use in render(image=...)."""
    if sum(bool(x) for x in (path, url, base64_data)) != 1:
        raise PipelineError("Give exactly one of path, url or base64_data.")
    if path:
        if not os.path.isfile(path):
            raise PipelineError("No file at " + path)
        data, name = open(path, "rb").read(), name or os.path.basename(path)
    elif url:
        if not url.lower().startswith(("http://", "https://")):
            raise PipelineError("Only http and https links.")
        req = urllib.request.Request(url, headers={"User-Agent": "opendaemon-comfyui-mcp/" + VERSION})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except Exception as e:
            raise PipelineError("Could not download that picture: " + str(e)[:200])
        with r:
            if not (r.headers.get("Content-Type") or "").startswith("image/"):
                raise PipelineError("That link is not a picture (Content-Type %s)." % r.headers.get("Content-Type"))
            data = r.read(25_000_001)
        name = name or os.path.basename(urllib.parse.urlparse(url).path) or "image.png"
    else:
        data = base64.b64decode(base64_data.split(",")[-1])
        name = name or "image.png"
    if len(data) > 25_000_000:
        raise PipelineError("Picture is bigger than 25 MB.")
    if not data[:12].startswith((b"\x89PNG", b"\xff\xd8\xff", b"RIFF", b"GIF8")):
        raise PipelineError("That file is not a PNG, JPG or WEBP picture.")
    name = "".join(c for c in os.path.basename(name) if c.isalnum() or c in "._- ")[:80] or "image.png"
    if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        name += ".png"
    r = api("/api/upload", {"name": name, "data": base64.b64encode(data).decode()})
    return {"image": r["name"], "note": "Use render(image=%r)." % r["name"]}


@mcp.tool(annotations=WEB)
def write_prompt(image: str, idea: str) -> dict:
    """Turns a short idea ("the cat waves hello") into a full video prompt, looking at the uploaded picture.
    Uses the AI keys set up in OpenDaemon (Gemini or NVIDIA). You can also write the prompt yourself."""
    return api("/api/genprompt", {"image": image, "idea": idea, "kind": "prompt"}, timeout=180)


@mcp.tool(annotations=WEB)
def write_avoid_list(image: str, prompt: str) -> dict:
    """Writes the avoid list (negative prompt) for a prompt. Only vace and LTX dev models use it."""
    return api("/api/genprompt", {"image": image, "idea": prompt, "kind": "negative"}, timeout=180)


@mcp.resource("guide://prompting", name="prompt-guide", mime_type="text/plain",
              description="How to use this server and write prompts that work.")
def prompt_guide() -> str:
    p = api("/api/presets")
    styles = "\n".join("  - %s: %s" % (s["name"], s["help"]) for s in p["styles"])
    quals = "\n".join("  - %s: %s" % (q["name"], q["help"]) for q in p["qualities"])
    return GUIDE + "\n\nStyles:\n" + styles + "\n\nQualities:\n" + quals + "\n\nCameras: " + ", ".join(p["cameras"])


@mcp.tool(annotations=READ)
def render_options() -> dict:
    """The models, styles, qualities, cameras and speeds render() accepts, with what each one does."""
    return api("/api/presets")


# ---------- D. render ----------
@mcp.tool(annotations=WRITE)
def render(prompt: str, image: str = "", model: str | None = None, style: str | None = None, quality: str | None = None,
           seconds: int | None = None, camera: str | None = None, speed: str | None = None, movement: int | None = None,
           seed: int = -1, avoid: str | None = None, width: int | None = None, height: int | None = None,
           priority: str = "user") -> dict:
    """Makes a video from a prompt, starting from a picture (image = a name from upload_image) or, with no
    image, from the prompt alone (text to video: describe the whole scene). Everything else is optional:
    model: small (default, fast) | ltx13 | vace | an added model's file name (list_models)
    style: e.g. "Balanced", "Soft and calm", "Lively", "Slow motion" (render_options lists all)
    quality: "Draft" (fastest) | "Normal" (default) | "High" | "Best" (1080p, very slow) | "Manual" (then width, height)
    seconds: 1-7 (2-3 holds together best). camera: still | zoom | zoomout | panr | panl. speed: slow | normal | fast
    movement: 10-70, how much things move (style sets it). seed: -1 = new each time; same seed + settings = same video.
    avoid: what must not appear (vace and LTX dev models only).
    priority: "user" (default, goes first) or "batch" (waits while the user's own renders go first).
    Returns a job id right away; then call job_status(id)."""
    a = _auto_args(locals())
    r = api("/api/render", {"auto": a, "priority": priority})
    return {"id": r["id"], "settings": r["settings"].get("easy"), "minutes": r["minutes"], "starts_in_min": r["starts_in_min"],
            "note": "Minutes are learned from this PC's own renders once it has a few. Call job_status(%r)." % r["id"]}


AUTO_KEYS = ("image", "prompt", "model", "style", "quality", "seconds", "camera", "speed", "movement", "seed", "avoid")


def _auto_args(v):
    a = {k: v.get(k) for k in AUTO_KEYS if k in v}
    if v.get("width") or v.get("height"):
        a.update(width=v.get("width") or 768, height=v.get("height") or 416)
    return a


@mcp.tool(annotations=WRITE)
def rerender(video: str, seed: int | None = None, prompt: str | None = None) -> dict:
    """Makes a video again with the settings of an earlier one (a name from list_videos).
    seed: leave out for the same seed (same video, e.g. after a crash), -1 for a new variation, or a number."""
    s = _settings_of(video)
    for k in ("render_seconds", "gpu", "review"):
        s.pop(k, None)
    if seed is not None:
        s["seed"] = seed
    if prompt:
        s["prompt"] = prompt
    r = api("/api/render", s)
    return {"id": r["id"], "note": "Call job_status(%r)." % r["id"]}


@mcp.tool(annotations=READ)
async def job_status(id: str = "", wait_seconds: int = 50, ctx: Context | None = None) -> dict:
    """Waits until the job is done (up to wait_seconds, max 55) and reports progress, then returns its status.
    Call again while status is queued or rendering. No id = the last 10 jobs.
    Also takes a batch id ("batch-...", from render_batch), a fix id ("fix-...", from auto_fix) or an edit id ("edit-...")."""
    if not id:
        return {"jobs": [_short_job(j) for j in (await asyncio.to_thread(api, "/api/jobs"))[:10]]}
    if id.startswith(("batch-", "fix-")):
        return await _wait_group(id, wait_seconds, ctx)
    if id.startswith("edit-"):
        end = time.time() + max(0, min(int(wait_seconds), 55))
        while True:
            e = await asyncio.to_thread(api, "/api/edit?id=" + urllib.parse.quote(id))
            if e["status"] != "running" or time.time() >= end:
                return _edit_answer(e)
            await asyncio.sleep(2)
    end, last = time.time() + max(0, min(int(wait_seconds), 55)), None
    while True:
        j = await asyncio.to_thread(api, "/api/job?id=" + urllib.parse.quote(id))
        p = j.get("progress")
        if ctx and p and j["status"] == "rendering" and (p["value"], p.get("pass_no")) != last:  # only when it moves
            last = (p["value"], p.get("pass_no"))
            await ctx.report_progress(p["value"], p["max"], "rendering, pass %d" % p.get("pass_no", 1))
        if j["status"] in ("done", "error", "cancelled") or time.time() >= end:
            return _short_job(j)
        await asyncio.sleep(2)


@mcp.tool(annotations=WRITE)
def cancel(id: str) -> dict:
    """Stops a queued or rendering job."""
    api("/api/cancel", {"id": id})
    return {"ok": True, "note": "Cancel sent; job_status shows cancelled within a few seconds."}


@mcp.tool(annotations=READ)
def diagnose_job(id: str) -> dict:
    """Why a job failed, in plain words, with what to try and the ComfyUI log lines."""
    return api("/api/diagnose?id=" + urllib.parse.quote(id))


# ---------- E. videos ----------
def _gallery():
    return api("/api/gallery")


def _settings_of(video):
    v = os.path.basename(video)
    for g in _gallery():
        if g["video"] == v:
            if not g["settings"]:
                raise PipelineError("No saved settings for " + v)
            return dict(g["settings"])
    raise PipelineError("No video called %r. Call list_videos." % v)


@mcp.tool(annotations=READ)
def list_videos(limit: int = 20) -> dict:
    """The finished videos, newest first, with their file path and short settings."""
    out = []
    for g in _gallery()[:max(1, min(limit, 200))]:
        s = g["settings"]
        out.append({"video": g["video"], "path": video_path(g["video"]), "settings": s.get("easy"),
                    "prompt": (s.get("prompt") or "")[:160], "seed": s.get("seed"), "image": s.get("image")})
    return {"folder": paths.OUT, "videos": out}


@mcp.tool(annotations=READ)
def video_info(video: str) -> dict:
    """Everything about one video: all its settings, size, length, path and a link to watch it."""
    s = _settings_of(video)
    p = video_path(video)
    return {"video": os.path.basename(p), "path": p, "url": PIPE + "/video/" + os.path.basename(p),
            "mb": round(os.path.getsize(p) / 1e6, 1) if os.path.exists(p) else None,
            "seconds": round((int(s.get("frames", 0)) - 1) / int(s.get("fps") or 24), 1) if s.get("frames") else None,
            "width": s.get("width"), "height": s.get("height"), "settings": s}


@mcp.tool(annotations=DESTROY)
def delete_video(video: str, user_confirmed: bool = False) -> dict:
    """Moves a video and its settings to the out/trash folder (not deleted for good). Ask the user first,
    then call with user_confirmed=true."""
    if not user_confirmed:
        return {"needs_confirmation": "This moves %s to the trash folder. Ask the user, then call again with user_confirmed=true." % video}
    return api("/api/delete", {"video": video})


# ---------- phase 2: mass creation, queue, self-check ----------
async def _wait_group(gid, wait_seconds, ctx):
    path = "/api/batch?id=" if gid.startswith("batch-") else "/api/fix?id="
    end, last = time.time() + max(0, min(int(wait_seconds), 55)), None
    while True:
        g = await asyncio.to_thread(api, path + urllib.parse.quote(gid))
        if gid.startswith("batch-"):
            n = len(g["jobs"])
            if ctx and (g["done"], g["failed"]) != last:
                last = (g["done"], g["failed"])
                await ctx.report_progress(g["done"] + g["failed"] + g["cancelled"], n, "%d of %d clips done" % (g["done"], n))
            out = {"id": gid, "status": g["status"], "label": g["label"], "done": g["done"], "failed": g["failed"],
                   "waiting_or_rendering": g["left"], "clips": [_short_job(j) for j in g["jobs"]]}
        else:
            if ctx and g.get("now") != last:  # MCP logging is deprecated: progress messages instead
                last = g.get("now")
                await ctx.report_progress(len(g["steps"]), None, g.get("now") or g["status"])
            out = {k: g.get(k) for k in ("id", "status", "now", "steps", "best", "best_score", "error")}
            if g.get("best"):
                out["best_path"] = video_path(g["best"])
        if out["status"] != "running" or time.time() >= end:
            return out
        await asyncio.sleep(3)


@mcp.tool(annotations=READ)
def estimate(image: str = "", prompt: str = "", model: str | None = None, style: str | None = None, quality: str | None = None,
             seconds: int | None = None, camera: str | None = None, speed: str | None = None, movement: int | None = None,
             width: int | None = None, height: int | None = None, items: list[dict] | None = None) -> dict:
    """How long a render (same settings as render) or a whole batch (items = a list of render settings) would take
    on this PC, and how long the queue is, WITHOUT starting anything."""
    if items:
        return api("/api/batch", {"items": items, "estimate_only": True})
    a = _auto_args(locals())
    a["image"] = a.get("image") or "_estimate_.png"
    a["prompt"] = a.get("prompt") or "x"
    r = api("/api/estimate", {"auto": a})
    return {"minutes": r["minutes"], "queue_wait_min": r["queue_wait_min"], "settings": r["settings"],
            "learned_from_renders": r["based_on_renders"]}


@mcp.tool(annotations=WRITE)
def render_batch(items: list[dict], label: str = "", priority: str = "batch", user_confirmed: bool = False) -> dict:
    """Many clips at once. items = a list of render settings, e.g. [{"image": "cat.png", "prompt": "...", "style": "Lively"}, ...]
    (same keys as render). All items are checked before any starts. Batches wait while the user's own renders go first.
    Over 30 minutes in total: tell the user the estimate and call again with user_confirmed=true.
    Returns a batch id; job_status(batch id) reports progress."""
    if not items:
        raise PipelineError("items is empty.")
    if len(items) > 100:
        raise PipelineError("At most 100 clips per batch.")
    est = api("/api/batch", {"items": items, "estimate_only": True})
    if est["minutes"] > 30 and not user_confirmed:
        return {"needs_confirmation": "%d clips, about %d minutes of rendering (plus %d min already queued). Ask the user, "
                                      "then call again with user_confirmed=true." % (est["clips"], est["minutes"], est["queue_wait_min"])}
    b = api("/api/batch", {"items": items, "label": label, "priority": priority})
    return {"id": b["id"], "clips": len(b["jobs"]), "minutes": b["minutes"], "note": "Call job_status(%r)." % b["id"]}


@mcp.tool(annotations=WRITE)
def render_variations(video: str, count: int = 3, vary: str = "seed", priority: str = "batch") -> dict:
    """Several versions of an earlier video to pick from (2-6). vary: "seed" (same settings, different luck: the usual
    choice) or "movement" (less and more movement around the original, same seed). Returns job ids for job_status."""
    if vary not in ("seed", "movement"):
        raise PipelineError('vary must be "seed" or "movement".')
    count = max(2, min(int(count), 6))
    s = _settings_of(video)
    for k in ("render_seconds", "gpu", "review"):
        s.pop(k, None)
    items = []
    for i in range(count):
        v = dict(s, seed=-1)
        if vary == "movement":
            comp = int(float(s.get("comp", 35)))
            v["comp"] = int(max(15, min(65, comp + (i - (count - 1) / 2) * 10)))
            v["seed"] = s.get("seed", -1)
        v["easy"] = "Variation %d/%d (%s) of %s" % (i + 1, count, vary, os.path.basename(video))
        items.append(v)
    ids = [api("/api/render", dict(v, _priority=priority))["id"] for v in items]
    return {"jobs": ids, "note": "Call job_status for each id, or job_status() for the list."}


@mcp.tool(annotations=WRITE)
def queue(action: str = "list", id: str = "", priority: str = "") -> dict:
    """The render queue. action: list (default) | pause (nothing new starts; the current render finishes) | resume |
    clear (cancels waiting jobs; priority="batch" clears only batch jobs) | front (move waiting job `id` first) |
    priority (set job `id` to priority "user" or "batch")."""
    if action == "list":
        return api("/api/queue")
    return api("/api/queue", {"action": action, "id": id or None, "priority": priority or None})


@mcp.tool(annotations=WEB)
def review_video(video: str, frames: int = 3, judge: bool = False) -> list:
    """Shows you the clip: frames spread over it (start, middle, end for 3) side by side in ONE picture, after the
    start picture, so you can SEE if it worked. judge=true also asks a vision model for a score and problems
    (uses the Gemini/NVIDIA key from OpenDaemon). Look for: the character changing, the action not happening,
    extra hands, melting, the end falling apart."""
    r = api("/api/review?" + urllib.parse.urlencode({"video": video, "n": frames}))
    pic = bool(r.get("start_image")) and os.path.isfile(r["start_image"])
    out = ["Video %s. Prompt: %s\n%s" % (r["video"], r["prompt"], "First picture: the START PICTURE. Second: frames of the video, left to right."
           if pic else "Made from the prompt alone (no start picture). The picture: frames of the video, left to right.")]
    if pic:
        out.append(Image(data=_small_jpeg(r["start_image"]), format="jpeg"))
    out.append(Image(data=open(r["sheet"], "rb").read(), format="jpeg"))
    if judge:
        v = api("/api/judge", {"video": video}, timeout=180)
        out.append("Judge (%s): score %s/10, problems: %s. %s" % (v.get("judge"), v["score"], ", ".join(v["problems"]) or "none", v["note"]))
    return out


def _small_jpeg(path):
    import io
    try:
        from PIL import Image as P
    except ImportError:
        return open(path, "rb").read()
    im = P.open(path).convert("RGB")
    im.thumbnail((640, 640))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return b.getvalue()


@mcp.tool(annotations=WRITE)
def auto_fix(video: str, problems: list[str] | None = None, tries: int = 2, priority: str = "user") -> dict:
    """The self-check loop for one video: a vision model judges it, the settings are fixed by the kind of problem
    (drift = less movement, no action = more movement, end melts = shorter, hands/melting = avoid list) and it renders
    again, at most `tries` times (1-3). Every try is kept; the answer names the best one.
    problems: if you already looked (review_video), say what you saw so the first judge call is skipped:
    drift, wrong_subject, no_action, too_much_motion, hands, melting, end_melts, talking, blurry.
    Returns a fix id; job_status(fix id) follows it."""
    c = api("/api/fix", {"video": video, "problems": problems or [], "tries": tries, "priority": priority})
    return {"id": c["id"], "note": "Each try is a full render. Call job_status(%r)." % c["id"]}


@mcp.tool(annotations=READ)
def suggest_settings(goal: str = "", model: str = "") -> dict:
    """Settings to start with for a goal ("calm portrait", "lively dance", "quick test"), plus the best-scored
    videos made on this PC so far (scores come from review_video(judge=true) and auto_fix)."""
    return api("/api/suggest?" + urllib.parse.urlencode({"goal": goal, "model": model}))


# ---------- phase 3: editing finished videos ----------
# Every edit makes a NEW video (the original is kept) and answers with its name; pass that name to the next edit.
# Videos are names from list_videos (or full paths). Quick edits answer at once; long ones return an "edit-..." id.
def _edit_answer(e):
    if e["status"] == "error":
        raise PipelineError(e["error"])
    if e["status"] == "running":
        return {"id": e["id"], "status": "running", "note": "Still working. Call job_status(%r)." % e["id"]}
    return dict(e["result"], status="done") if isinstance(e["result"], dict) else {"status": "done", "result": e["result"]}


def _edit(op, **args):
    return _edit_answer(api("/api/edit", {"op": op, "args": {k: v for k, v in args.items() if v is not None}}, timeout=90))


@mcp.tool(annotations=WRITE)
def build_video(clips: list, transition: str = "crossfade", shape: str | None = None, voice: str | dict | None = None,
                captions: str | bool | None = None, music: str | dict | None = None, title: str | None = None,
                end_card: str | bool | None = None, export: str | None = None) -> dict:
    """ONE call for a finished video. clips: video names in order, or {"video", "start", "end", "speed"} to cut/speed one.
    transition: cut | fade | crossfade. shape: "9:16" (Shorts/Reels/TikTok) | "1:1" | "4:5" | "16:9".
    voice: the words to say (or {"text", "voice": andrew|emma|ava|brian|es|pt..., "engine": edge|local, "sample"} or {"audio": file}).
    captions: "auto" (timed to the voice) or your own text. music: a music file path (or {"file", "volume"}); it gets quieter under the voice.
    title: text for a card at the start. end_card: text for the last card (true = "Follow for more").
    export: shorts | reels | tiktok | facebook | youtube | x | square (right size and format to post).
    Only the final video stays in the gallery."""
    return _edit("build_video", clips=clips, transition=transition, shape=shape, voice=voice, captions=captions, music=music,
                 title=title, end_card=end_card, export=export)


@mcp.tool(annotations=WRITE)
def join_clips(videos: list[str], transition: str = "cut", seconds: float = 0.5) -> dict:
    """Clips into one video, in order. transition: cut | fade (through black) | crossfade; seconds = its length."""
    return _edit("join", videos=videos, transition=transition, seconds=seconds)


@mcp.tool(annotations=WRITE)
def trim_clip(video: str, start: float = 0.0, end: float | None = None) -> dict:
    """Keeps only start..end seconds. Good for cutting the end of a clip where it starts to melt."""
    return _edit("trim", v=video, start=start, end=end)


@mcp.tool(annotations=WRITE)
def extend_clip(video: str, prompt: str, seconds: int | None = None, transition: str = "cut") -> dict:
    """GPU. Makes a video longer without it falling apart: the last frame becomes the start of a new clip
    (same model and settings, your prompt for what happens next), then both are joined. Takes a render's time."""
    return _edit("extend_clip", video=video, prompt=prompt, seconds=seconds, transition=transition)


@mcp.tool(annotations=WRITE)
def change_speed(video: str, speed: float) -> dict:
    """0.5 = half speed (slow motion), 2 = twice as fast. 0.1 to 8."""
    return _edit("speed", v=video, x=speed)


@mcp.tool(annotations=WRITE)
def reverse_clip(video: str) -> dict:
    """Plays the clip backwards."""
    return _edit("reverse", v=video)


@mcp.tool(annotations=WRITE)
def loop_clip(video: str, times: int = 2) -> dict:
    """Repeats the clip (2-20 times)."""
    return _edit("loop", v=video, times=times)


@mcp.tool(annotations=WRITE)
def boomerang(video: str, times: int = 1) -> dict:
    """Forwards then backwards, like a boomerang. times = how many back-and-forths."""
    return _edit("boomerang", v=video, times=times)


@mcp.tool(annotations=WRITE)
def smooth_motion(video: str, fps: int = 48) -> dict:
    """Smoother motion: makes in-between frames, 24 -> 30/48/50/60 fps (on the processor, about 2 s per second of video)."""
    return _edit("smooth_motion", v=video, fps=fps)


@mcp.tool(annotations=WRITE)
def upscale_clip(video: str, method: str = "simple", height: int = 1080) -> dict:
    """Bigger video. method "simple": resize to `height` with sharpening (seconds, any video). method "ltx": GPU,
    renders the same clip again with the LTX second pass to 1080p - real detail, but 30-50 min on a 4 GB card."""
    if method == "ltx":
        return _edit("upscale_ltx", video=video)
    return _edit("upscale_simple", v=video, height=height)


@mcp.tool(annotations=WRITE)
def crop_zoom(video: str, zoom: float = 1.25, direction: str = "in", x: float = 0.5, y: float = 0.5) -> dict:
    """A slow camera move on a finished clip. direction: in | out | left | right | up | down | none (a still crop).
    zoom 1.01-4. x, y (0-1) = the point to zoom on (0.5, 0.5 = the middle)."""
    return _edit("crop_zoom", v=video, zoom=zoom, direction=direction, x=x, y=y)


@mcp.tool(annotations=WRITE)
def color_adjust(video: str, brightness: float = 0.0, contrast: float = 1.0, saturation: float = 1.0, warmth: float = 0.0) -> dict:
    """brightness -1..1 (0 = same), contrast 0.5..2 (1 = same), saturation 0..3 (1 = same), warmth -1 cool .. 1 warm."""
    return _edit("color_adjust", v=video, brightness=brightness, contrast=contrast, saturation=saturation, warmth=warmth)


@mcp.tool(annotations=WRITE)
def fade(video: str, fade_in: float = 0.5, fade_out: float = 0.5) -> dict:
    """Fades from black at the start and to black at the end (seconds; 0 = none). Sound fades too."""
    return _edit("fade", v=video, fade_in=fade_in, fade_out=fade_out)


@mcp.tool(annotations=WRITE)
def add_music(video: str, music: str, volume: float = 0.25, duck: bool = True) -> dict:
    """Background music (a file path), looped or cut to the video's length, faded in and out.
    If the video has a voice, the music gets quieter while it speaks (duck)."""
    return _edit("add_music", v=video, music=music, volume=volume, duck=duck)


@mcp.tool(annotations=WEB)
def make_voice(text: str, voice: str = "andrew", engine: str = "edge", sample: str | None = None, language: str = "en") -> dict:
    """A voice line from text, as an audio file for add_voice. engine "edge" (default): free online voices, fast,
    with word timings for auto captions; voice: andrew, emma, ava, brian, es, es-f, pt, pt-f, or any edge voice name.
    engine "local": Chatterbox on this PC, offline, CLONES the voice in `sample` (a 5-15 s recording); slow."""
    return _edit("make_voice", text=text, voice=voice, engine=engine, sample=sample, language=language)


@mcp.tool(annotations=WRITE)
def add_voice(video: str, audio: str, start: float = 0.3, volume: float = 1.0, keep_original: bool = False) -> dict:
    """Puts a voice line (from make_voice, or any sound file) on a clip, starting at `start` seconds.
    If it is longer than the clip, the last frame is held until it ends."""
    return _edit("add_voice", v=video, audio=audio, start=start, volume=volume, keep_original=keep_original)


@mcp.tool(annotations=WRITE)
def lipsync(video: str, audio: str) -> dict:
    """Moves a real human face's lips to a voice (LatentSync). Needs a graphics card with 8 GB or more."""
    return _edit("lipsync", video=video, audio=audio)


@mcp.tool(annotations=WRITE)
def add_captions(video: str, text: str = "auto", style: str = "bold", position: str = "bottom", highlight: bool = False,
                 words_per_caption: int = 4) -> dict:
    """Burned-in captions. text "auto" = timed to the voice line put on with add_voice (edge voices). Or your own text,
    spread over the clip ("|" forces a break). style: bold | impact | clean | box. position: bottom | lower | center | top.
    highlight = yellow text."""
    return _edit("add_captions", v=video, text=text, style=style, position=position, highlight=highlight,
                 words_per_caption=words_per_caption)


@mcp.tool(annotations=WRITE)
def add_text(video: str, text: str, position: str = "top", start: float = 0.0, end: float | None = None, style: str = "bold",
             box: bool = False) -> dict:
    """A title or any text on the video, from `start` to `end` seconds (default: the whole clip)."""
    return _edit("add_text", v=video, text=text, position=position, start=start, end=end, style=style, box=box)


@mcp.tool(annotations=WRITE)
def add_logo(video: str, image: str, position: str = "top-right", scale: float = 0.15, opacity: float = 0.9) -> dict:
    """A logo or watermark (a picture file, PNG with transparency is best). position: top-right | top-left |
    bottom-right | bottom-left | center. scale = its width as part of the video's width."""
    return _edit("add_logo", v=video, image=image, position=position, scale=scale, opacity=opacity)


@mcp.tool(annotations=WRITE)
def title_card(video: str, text: str = "", image: str | None = None, seconds: float = 2.0) -> dict:
    """A card at the START: text and/or a picture on a dark background, faded into the video."""
    return _edit("title_card", v=video, text=text, image=image, seconds=seconds)


@mcp.tool(annotations=WRITE)
def end_card(video: str, text: str = "Follow for more", image: str | None = None, seconds: float = 2.5) -> dict:
    """A card at the END ("Follow for more", your logo...), faded in after the video."""
    return _edit("end_card", v=video, text=text, image=image, seconds=seconds)


@mcp.tool(annotations=WRITE)
def reformat(video: str, shape: str = "9:16", mode: str = "blur") -> dict:
    """Changes the shape: 9:16 (Shorts, Reels, TikTok) | 1:1 | 4:5 | 16:9. mode: blur (blurred copy behind, nothing
    is cut off) | crop (fills the frame, cuts the sides) | pad (black bars)."""
    return _edit("reformat", v=video, shape=shape, mode=mode)


@mcp.tool(annotations=WRITE)
def export(video: str, platform: str = "shorts", max_mb: float | None = None) -> dict:
    """Ready to post: the platform's shape, H.264 + AAC, file size under the limit.
    platform: shorts | reels | tiktok | facebook (10 MB) | youtube | x | square. max_mb overrides the size limit."""
    return _edit("export", v=video, platform=platform, max_mb=max_mb)


@mcp.tool(annotations=WRITE)
def thumbnail(video: str, at: float | None = None) -> dict:
    """A still picture (JPG) from the video at `at` seconds (default: the middle), for a post's cover."""
    return _edit("thumbnail", v=video, at=at)


@mcp.tool(annotations=WRITE)
def make_gif(video: str, fps: int = 12, width: int = 480) -> dict:
    """An animated GIF of the video."""
    return _edit("make_gif", v=video, fps=fps, width=width)


@mcp.tool(annotations=WRITE)
def extract_audio(video: str) -> dict:
    """Saves the video's sound as an .m4a file."""
    return _edit("extract_audio", v=video)


@mcp.tool(annotations=WRITE)
def remove_audio(video: str) -> dict:
    """The same video without sound."""
    return _edit("remove_audio", v=video)


# ---------- G. your own ComfyUI workflows ----------
@mcp.tool(annotations=READ)
def list_workflows() -> dict:
    """Your own ComfyUI workflows (API-format .json files in the workflows folder) and the inputs each one lets
    you change (prompt, image, seed, steps... found automatically, or nodes titled "$name" in ComfyUI)."""
    return api("/api/workflows")


@mcp.tool(annotations=WRITE)
def run_workflow(name: str, inputs: dict | None = None, priority: str = "user", keep_seed: bool = False,
                 force: bool = False) -> dict:
    """Runs one of your workflows. inputs = {"prompt": "...", "image": "C:/pics/cat.png", "steps": 20} using the
    names from list_workflows; a file path for image/video/audio is copied into ComfyUI for you. A new random seed
    each run unless you pass seed or keep_seed=true. Checks the workflow first and refuses if it would fail
    (force=true runs anyway). Returns a job id: wait with job_status; the saved files come back in "files"."""
    return api("/api/workflow/run", {"name": name, "inputs": inputs or {}, "priority": priority,
                                     "keep_seed": keep_seed, "force": force}, timeout=120)


@mcp.tool(annotations=READ)
def check_workflow(name: str = "", workflow: dict | None = None) -> dict:
    """Before a slow run: finds missing node types (and which node pack has them), missing model files and wrong
    values in a saved workflow (name) or one you give (workflow, API format). Also lists its changeable inputs."""
    if not name and not workflow:
        raise PipelineError("Give name (a saved workflow) or workflow (the API-format JSON).")
    if name:
        return api("/api/workflow/check?name=" + urllib.parse.quote(name)) if not workflow else \
            api("/api/workflow/check", {"name": name})
    return api("/api/workflow/check", {"workflow": workflow}, timeout=120)


@mcp.tool(annotations=WEB)
def install_custom_node(source: str, user_confirmed: bool = False) -> dict:
    """Installs a ComfyUI node pack: a GitHub link, a pack name, or a node type from check_workflow. Node packs
    are programs from other people, so ASK THE USER FIRST, then call again with user_confirmed=true.
    ComfyUI needs a restart after (restart_comfyui)."""
    return api("/api/workflow/install", {"source": source, "user_confirmed": user_confirmed}, timeout=2400)


@mcp.tool(annotations=WRITE)
def save_workflow(name: str, workflow: dict, overwrite: bool = False) -> dict:
    """Keeps a workflow (ComfyUI API format) under a name, so run_workflow can use it. Returns its inputs."""
    return api("/api/workflow/save", {"name": name, "workflow": workflow, "overwrite": overwrite})


# ---------- which tools to show ----------
# Every tool's description and settings are sent to the model with EVERY message (~8k tokens for all 62).
# Chat apps that add MCP tools to all chats (OpenDaemon does) should use the small "core" set.
CORE = ["comfyui_status", "start_comfyui", "list_models", "upload_image", "write_prompt", "render", "job_status", "cancel",
        "list_videos", "review_video", "auto_fix", "render_batch", "build_video", "join_clips", "trim_clip"]


def _compact(node, in_props=False):
    """Smaller schemas: drop the per-field "title" labels and turn "X or null" into plain X (null still works)."""
    if isinstance(node, list):
        return [_compact(x) for x in node]
    if not isinstance(node, dict):
        return node
    if in_props:  # keys here are parameter NAMES (one is even called "title"): keep them all
        return {k: _compact(v) for k, v in node.items()}
    out = {}
    for k, v in node.items():
        if k == "title" and isinstance(v, str):
            continue
        out[k] = _compact(v, in_props=(k == "properties"))
    any_of = out.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2 and {"type": "null"} in any_of:
        out.pop("anyOf")
        out.update(next(x for x in any_of if x != {"type": "null"}))
    return out


def apply_toolset(which=None):
    which = (which or os.environ.get("OPENDAEMON_COMFY_TOOLS") or "full").strip()
    tm = mcp._tool_manager
    if which != "full":
        keep = set(CORE if which == "core" else [x.strip() for x in which.split(",") if x.strip()])
        for t in list(tm.list_tools()):
            if t.name not in keep:
                mcp.remove_tool(t.name)
    for t in tm.list_tools():
        t.parameters = _compact(t.parameters)


apply_toolset()


def main():
    mcp.run()


if __name__ == "__main__":
    main()
