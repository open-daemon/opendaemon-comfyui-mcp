"""Your own ComfyUI workflows (tool group G).

A workflow is ComfyUI's API format: {"<node id>": {"class_type": ..., "inputs": {...}, "_meta": {"title": ...}}}.
In ComfyUI: Workflow menu > Export (API). The normal "Save" format (with "nodes" and "links") is for the page
and can't be run over the API, so we say so instead of guessing.

The inputs an agent can change are found two ways:
  - automatically: prompt / negative (text boxes, told apart by which sampler input they feed), image, video,
    audio, seed (every seed in the graph moves together), steps, cfg, denoise, width, height, length, fps;
  - by a title marker, which wins: name a node "$name" (its main input) or "$name.field" in ComfyUI.
    Several nodes with the same "$name" move together (one value sets them all).
"""
import copy
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request

import paths

HERE = os.path.dirname(os.path.abspath(__file__))
WF_DIR = os.environ.get("COMFY_WORKFLOWS_DIR") or os.path.join(paths.DATA, "workflows")
CACHE = os.path.join(paths.DATA, "cache")
NODE_MAP_URL = "https://raw.githubusercontent.com/ltdrdata/ComfyUI-Manager/main/extension-node-map.json"
MODEL_EXT = (".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin", ".sft", ".onnx")
MEDIA_EXT = (".mp4", ".webm", ".mov", ".mkv", ".gif", ".webp", ".png", ".jpg", ".jpeg", ".wav", ".mp3", ".flac", ".m4a")

# field -> param name, for the auto inputs. Order matters for the list shown to the agent.
NUMBERS = {"steps": "steps", "cfg": "cfg", "denoise": "denoise", "width": "width", "height": "height",
           "length": "length", "num_frames": "length", "frames": "length", "frame_count": "length",
           "fps": "fps", "frame_rate": "fps", "batch_size": "batch_size"}
SEEDS = ("seed", "noise_seed")
MAIN_FIELDS = ("text", "prompt", "image", "video", "audio", "value", "string", "seed", "noise_seed", "int", "float")


class WorkflowError(ValueError):
    pass


# ---------- files ----------
def seed_examples():
    """The first time, the workflows folder gets our example workflows, so there is something to run."""
    if os.path.isdir(WF_DIR):
        return
    os.makedirs(WF_DIR, exist_ok=True)
    src = os.path.join(HERE, "examples", "workflows")
    for f in os.listdir(src) if os.path.isdir(src) else []:
        if f.endswith(".json"):
            shutil.copy2(os.path.join(src, f), os.path.join(WF_DIR, f))


def _safe(name):
    n = os.path.basename(str(name or "")).strip()
    if n.lower().endswith(".json"):
        n = n[:-5]
    if not n or not re.fullmatch(r"[\w .()\-]+", n):
        raise WorkflowError("Workflow names use letters, numbers, spaces, - _ . ( ) only. Got %r." % name)
    return n


def path_of(name):
    return os.path.join(WF_DIR, _safe(name) + ".json")


def is_api(g):
    return isinstance(g, dict) and bool(g) and all(isinstance(v, dict) and "class_type" in v for v in g.values())


def as_graph(data):
    """Accepts the API format (as a dict or JSON text). Refuses the page format with how to fix it."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except ValueError as e:
            raise WorkflowError("That isn't valid JSON: %s" % e)
    if isinstance(data, dict) and isinstance(data.get("prompt"), dict) and is_api(data["prompt"]):
        data = data["prompt"]  # a /prompt request body or a history entry
    if isinstance(data, dict) and "nodes" in data and "links" in data:
        raise WorkflowError("This is ComfyUI's page format, which can't be run over the API. In ComfyUI open the "
                            "workflow, then Workflow menu > Export (API), and use that file.")
    if not is_api(data):
        raise WorkflowError("Not a ComfyUI API workflow: expected {\"<node id>\": {\"class_type\": ..., \"inputs\": {...}}}.")
    return data


def load(name):
    seed_examples()
    p = path_of(name)
    if not os.path.isfile(p):
        have = ", ".join(sorted(x[:-5] for x in os.listdir(WF_DIR) if x.endswith(".json"))) if os.path.isdir(WF_DIR) else ""
        raise WorkflowError("No workflow called %r. %s" % (name, ("Saved: " + have) if have else
                            "None saved yet: put API-format .json files in " + WF_DIR + " or use save_workflow."))
    try:
        return as_graph(open(p, encoding="utf-8").read())
    except WorkflowError as e:
        raise WorkflowError("%s: %s" % (os.path.basename(p), e))


def save(name, data, overwrite=False):
    g = as_graph(data)
    p = path_of(name)
    if os.path.exists(p) and not overwrite:
        raise WorkflowError("A workflow called %r already exists. Pass overwrite=true to replace it." % _safe(name))
    os.makedirs(WF_DIR, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(g, f, indent=1)
    return {"name": _safe(name), "path": p, "nodes": len(g), "inputs": params(g)}


# ---------- the inputs an agent can change ----------
def _is_link(v):
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], (str, int)) and isinstance(v[1], int)


def _kind(v):
    return "bool" if isinstance(v, bool) else "int" if isinstance(v, int) else "float" if isinstance(v, float) else "text"


def _negatives(g):
    """Node ids whose output goes into an input called negative (directly or through one conditioning node)."""
    direct = {str(v[0]) for n in g.values() for k, v in n.get("inputs", {}).items() if "negative" in k.lower() and _is_link(v)}
    out = set(direct)
    for nid in direct:  # e.g. text -> ConditioningZeroOut / FluxGuidance -> negative
        for k, v in g.get(nid, {}).get("inputs", {}).items():
            if _is_link(v) and k.lower() in ("conditioning", "conditioning_to", "conditioning_from"):
                out.add(str(v[0]))
    return out


def params(g):
    """[{name, type, value, targets: [[node, field], ...], how}] in a stable order."""
    found, used = [], set()

    def add(name, nid, field, how):
        v = g[nid]["inputs"].get(field)
        for p in found:
            if p["name"] == name:
                if name == "seed" or (how == "title" and p["how"] == "title"):  # every seed moves together; so do nodes with the same "$name"
                    p["targets"].append([nid, field])
                    return
                i = 2
                while any(q["name"] == "%s_%d" % (name, i) for q in found):
                    i += 1
                name = "%s_%d" % (name, i)
                break
        found.append({"name": name, "type": _kind(v), "value": v, "targets": [[nid, field]], "how": how,
                      "node": "%s (%s)" % (g[nid].get("_meta", {}).get("title") or g[nid]["class_type"], nid)})
        used.add((nid, field))

    order = sorted(g, key=lambda k: (int(k) if str(k).isdigit() else 10 ** 9, str(k)))
    for nid in order:  # 1) title markers
        title = str(g[nid].get("_meta", {}).get("title") or "")
        m = re.match(r"^\$([A-Za-z_]\w*)(?:\.(\w+))?", title.strip())
        if not m:
            continue
        ins = g[nid].get("inputs", {})
        field = m.group(2) or next((f for f in MAIN_FIELDS if f in ins and not _is_link(ins[f])),
                                   next((f for f, v in ins.items() if not _is_link(v)), None))
        if field and field in ins and not _is_link(ins[field]):
            add(m.group(1), nid, field, "title")
    neg = _negatives(g)
    for nid in order:  # 2) automatic
        n = g[nid]
        ins, ct = n.get("inputs", {}), n.get("class_type", "")
        free = {f: v for f, v in ins.items() if not _is_link(v) and (nid, f) not in used}
        if any(t[0] == nid for p in found for t in p["targets"] if p["how"] == "title"):
            continue
        for f in ("text", "prompt"):
            if isinstance(free.get(f), str) and ("Text" in ct or "Prompt" in ct or "CLIP" in ct or "Encode" in ct):
                add("negative" if nid in neg else "prompt", nid, f, "auto")
                break
        if ct.startswith("Load") or "LoadImage" in ct or "LoadVideo" in ct or "LoadAudio" in ct:
            for f, name in (("image", "image"), ("video", "video"), ("audio", "audio"), ("file", "video")):
                if isinstance(free.get(f), str):
                    add(name, nid, f, "auto")
                    break
        for f in SEEDS:
            if isinstance(free.get(f), int) and not isinstance(free.get(f), bool):
                add("seed", nid, f, "auto")
        for f, name in NUMBERS.items():
            if f in free and isinstance(free[f], (int, float)) and not isinstance(free[f], bool):
                if f == "batch_size" and "Latent" not in ct:
                    continue
                add(name, nid, f, "auto")
    return found


def describe(g):
    return [{"name": p["name"], "type": p["type"], "now": p["value"], "node": p["node"]} for p in params(g)]


def apply(g, inputs=None, keep_seed=False):
    """A copy of the graph with the inputs set. A missing or -1 seed becomes a new random one unless keep_seed."""
    g = copy.deepcopy(g)
    ps = {p["name"].lower(): p for p in params(g)}
    used = {}
    for k, v in (inputs or {}).items():
        p = ps.get(str(k).lower())
        if not p:
            raise WorkflowError("This workflow has no input %r. Its inputs: %s." % (k, ", ".join(sorted(ps)) or "none found"))
        try:
            v = (bool(v) if p["type"] == "bool" else int(round(float(v))) if p["type"] == "int"
                 else float(v) if p["type"] == "float" else str(v))
        except (TypeError, ValueError):
            raise WorkflowError("%s must be a %s, got %r." % (p["name"], p["type"], v))
        for nid, f in p["targets"]:
            g[nid]["inputs"][f] = v
        used[p["name"]] = v
    seed = ps.get("seed")
    if seed and not keep_seed and ("seed" not in {k.lower() for k in (inputs or {})} or used.get("seed", 0) < 0):
        s = random.randint(0, 2 ** 31 - 1)
        for nid, f in seed["targets"]:
            g[nid]["inputs"][f] = s
        used["seed"] = s
    return g, used


# ---------- check against the ComfyUI that is running ----------
def _choices(spec):
    if not isinstance(spec, (list, tuple)) or not spec:
        return None
    if isinstance(spec[0], list):
        return spec[0]
    if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("options")
    return None


def node_map():
    """{node class: [node pack repo urls]} from ComfyUI-Manager's public list, cached for a week. {} when offline."""
    p = os.path.join(CACHE, "extension-node-map.json")
    try:
        if not os.path.exists(p) or time.time() - os.path.getmtime(p) > 7 * 86400:
            req = urllib.request.Request(NODE_MAP_URL, headers={"User-Agent": "opendaemon-comfyui-mcp"})
            with urllib.request.urlopen(req, timeout=20) as r:
                data = r.read()
            json.loads(data)
            os.makedirs(CACHE, exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
        raw = json.load(open(p, encoding="utf-8"))
    except Exception:
        return {}
    out = {}
    for url, v in raw.items():
        for cls in (v[0] if isinstance(v, list) and v else []):
            out.setdefault(cls, []).append(url)
    return out


def check(g, info):
    """Problems that would make the run fail, found before it spends minutes on the GPU. info = ComfyUI /object_info."""
    problems, missing_nodes, missing_files = [], [], []
    for nid, n in g.items():
        ct = n.get("class_type")
        spec = info.get(ct)
        if spec is None:
            missing_nodes.append(ct)
            continue
        req = spec.get("input", {}).get("required", {}) or {}
        opt = spec.get("input", {}).get("optional", {}) or {}
        ins = n.get("inputs", {})
        for f in req:
            if f not in ins:
                problems.append("Node %s (%s) is missing its input %r." % (nid, ct, f))
        for f, v in ins.items():
            if _is_link(v):
                if str(v[0]) not in g:
                    problems.append("Node %s (%s): input %r is connected to node %s, which isn't in the workflow." % (nid, ct, f, v[0]))
                continue
            ch = _choices(req.get(f) or opt.get(f))
            if ch is not None and v not in ch and not isinstance(v, (int, float)):
                if isinstance(v, str) and v.lower().endswith(MODEL_EXT):
                    missing_files.append({"file": v, "node": "%s (%s)" % (ct, nid), "input": f})
                elif isinstance(v, str) and v.lower().endswith(MEDIA_EXT):
                    problems.append("Node %s (%s): %r is not in ComfyUI's input folder. Upload it or pass the input." % (nid, ct, v))
                else:
                    problems.append("Node %s (%s): %r is not a valid %s. Some valid ones: %s." % (
                        nid, ct, v, f, ", ".join(map(str, ch[:8]))))
    packs = {}
    if missing_nodes:
        nm = node_map()
        for c in sorted(set(missing_nodes)):
            packs[c] = nm.get(c, [])[:3]
    out = {"ok": not (problems or missing_nodes or missing_files), "nodes": len(g), "problems": problems,
           "missing_nodes": [{"node": c, "install_from": packs.get(c) or []} for c in sorted(set(missing_nodes))],
           "missing_model_files": missing_files}
    if missing_nodes:
        out["how_to_fix_nodes"] = ("Install the node pack with install_custom_node (asks the user first), then restart ComfyUI."
                                   if any(packs.values()) else "These node types aren't in ComfyUI-Manager's list; find the node pack by name.")
    if missing_files:
        out["how_to_fix_files"] = "Download the model files (search_models / download_model) or add them with add_model_file."
    return out


def outputs(entry):
    """Files a finished run saved, from ComfyUI's /history entry (output folder only, not previews)."""
    files = []
    for nid, o in (entry.get("outputs") or {}).items():
        for key in ("videos", "gifs", "images", "audio", "files"):
            for it in o.get(key) or []:
                if not isinstance(it, dict) or not it.get("filename") or it.get("type", "output") != "output":
                    continue
                files.append(os.path.join(paths.COMFY, "output", it.get("subfolder") or "", it["filename"]))
    seen, out = set(), []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            out.append(f)
    return out


# ---------- node packs ----------
def _repo_url(source):
    s = str(source or "").strip()
    if re.fullmatch(r"https://github\.com/[\w.\-]+/[\w.\-]+(\.git)?/?", s):
        return s.rstrip("/").removesuffix(".git")
    nm = node_map()
    if s in nm:  # a node class name
        return nm[s][0].rstrip("/").removesuffix(".git") if nm[s] else None
    want = s.lower()
    for urls in nm.values():
        for u in urls:
            if u.rstrip("/").lower().endswith("/" + want):
                return u.rstrip("/").removesuffix(".git")
    return None


def install(source, user_confirmed=False):
    url = _repo_url(source)
    if not url or not url.startswith("https://github.com/"):
        raise WorkflowError("Can't find a node pack for %r. Give its GitHub link (https://github.com/owner/repo) "
                            "or a node type name from check_workflow." % source)
    name = url.rsplit("/", 1)[1]
    dst = os.path.join(paths.COMFY, "custom_nodes", name)
    if os.path.exists(dst):
        return {"ok": True, "installed": False, "note": "%s is already installed (%s). Restart ComfyUI if it isn't loaded." % (name, dst)}
    plan = {"pack": name, "from": url, "to": dst,
            "will": "download the code with git, then install its Python requirements into ComfyUI's Python. "
                    "Node packs are programs: only install ones you trust. ComfyUI needs a restart after."}
    if not user_confirmed:
        return dict(plan, needs_confirmation=True,
                    note="Ask the user. If they agree, call install_custom_node again with user_confirmed=true.")
    git = shutil.which("git")
    if not git:
        raise WorkflowError("git isn't installed, so the node pack can't be downloaded. Install Git (git-scm.com) and try again.")
    r = subprocess.run([git, "clone", "--depth", "1", url + ".git", dst], capture_output=True, text=True, timeout=600)
    if r.returncode:
        shutil.rmtree(dst, ignore_errors=True)
        raise WorkflowError("Downloading %s failed: %s" % (name, (r.stderr or r.stdout)[-400:]))
    note = "Installed. Restart ComfyUI (restart_comfyui) so it loads the new nodes."
    reqs = os.path.join(dst, "requirements.txt")
    py = paths.comfy_python()
    if os.path.isfile(reqs):
        if not py:
            note = "Downloaded, but ComfyUI's Python wasn't found, so its requirements weren't installed (set COMFYUI_PYTHON). " + note
        else:
            r = subprocess.run([py, "-s", "-m", "pip", "install", "-r", reqs], capture_output=True, text=True, timeout=1800)
            if r.returncode:
                note = "Downloaded, but installing its requirements failed: %s. %s" % ((r.stderr or r.stdout)[-400:], note)
    return dict(plan, ok=True, installed=True, note=note)
