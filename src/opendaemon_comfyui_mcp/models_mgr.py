"""Model manager for the video pipeline: one-click downloads, "add your own model", and a file check
that says in plain words what a model file is, or why this app cannot run it.

We never host or bundle models: downloads come straight from the makers' Hugging Face pages.
"""
import json
import os
import shutil
import struct
import subprocess
import threading
import time
import urllib.request

import paths
COMFY = paths.COMFY
MODELS = os.path.join(COMFY, "models")
HERE = os.path.dirname(os.path.abspath(__file__))
CUSTOM = os.path.join(paths.DATA, "custom-models.json")
HF = "https://huggingface.co/"
GB = 1e9

# Files shared by several models. Sizes are the real byte counts (checked 2026-11-18).
T5 = {"url": HF + "comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors",
      "folder": "text_encoders", "file": "t5xxl_fp8_e4m3fn.safetensors", "size": 4.89 * GB}
LTX_SMALL = {"url": HF + "Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled-fp8.safetensors",
             "folder": "checkpoints", "file": "ltxv-2b-0.9.8-distilled-fp8.safetensors", "size": 4.46 * GB}
UMT5 = {"url": HF + "city96/umt5-xxl-encoder-gguf/resolve/main/umt5-xxl-encoder-Q4_K_M.gguf",
        "folder": "text_encoders", "file": "umt5-xxl-encoder-Q4_K_M.gguf", "size": 3.66 * GB}
WAN_VAE = {"url": HF + "Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors",
           "folder": "vae", "file": "wan_2.1_vae.safetensors", "size": 0.25 * GB}


def _ltx13(q, size, card, tested=False):
    f = "LTXV-13B-0.9.8-distilled-%s.gguf" % q
    # rank: "Best for your card" = the highest rank whose card size fits. LTX 13B is David's best model by far,
    # so its biggest version that fits wins.
    return {"id": "ltx13-" + q.lower(), "name": "LTX-Video 13B Distilled (GGUF %s)" % q, "type": "ltx13",
            "rank": {"Q3_K_M": 10, "Q4_K_M": 20, "Q5_K_M": 30, "Q8_0": 40}[q],
            "card": card, "tested": tested, "builtin": "ltx13" if q == "Q3_K_M" else None,
            "help": "Sharper than the 2B and keeps the character. Bigger number = better quality, needs more memory.",
            "files": [{"url": HF + "QuantStack/LTXV-13B-0.9.8-distilled-GGUF/resolve/main/" + f,
                       "folder": "unet", "file": f, "size": size * GB}, LTX_SMALL, T5]}


def _vace14(q, size, card):
    f = "Wan2.1_14B_VACE-%s.gguf" % q
    return {"id": "vace14-" + q.lower(), "name": "Wan 2.1 VACE 14B (GGUF %s)" % q, "type": "vace", "card": card,
            "help": "The big Wan. Much better motion than the 1.3B, but slow: made for strong cards.",
            "files": [{"url": HF + "QuantStack/Wan2.1_14B_VACE-GGUF/resolve/main/" + f,
                       "folder": "unet", "file": f, "size": size * GB}, UMT5, WAN_VAE]}


CATALOG = [
    {"id": "ltx2b-distilled", "name": "LTX-Video 2B Distilled (fp8)", "type": "ltxdist", "card": 4, "tested": True,
     "builtin": "small", "help": "Fast, about 6 min on a low-end GTX 1050 Ti 4 GB; a better card is faster. What all our good clips used.",
     "files": [LTX_SMALL, T5]},
    # LTX-Video 2B 0.9.6 Dev removed 11-18: distorted pictures in David's tests; the distilled one is better.
    _ltx13("Q3_K_M", 6.51, 4, tested=True),
    _ltx13("Q4_K_M", 8.82, 8),
    _ltx13("Q5_K_M", 9.82, 10),
    _ltx13("Q8_0", 14.01, 16),
    {"id": "vace13", "name": "Wan 2.1 VACE 1.3B (official)", "type": "vace", "card": 4, "tested": True, "builtin": "vace",
     "help": "Small Wan, 16 fps. Keeps your picture as the first frame.",
     "files": [{"url": HF + "Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_vace_1.3B_fp16.safetensors",
                "folder": "diffusion_models", "file": "wan2.1_vace_1.3B_fp16.safetensors", "size": 4.31 * GB}, UMT5, WAN_VAE]},
    _vace14("Q3_K_S", 7.84, 8),
    _vace14("Q4_K_M", 11.64, 12),
    _vace14("Q8_0", 18.66, 24),
    {"id": "ltx-upscaler", "name": "LTX upscaler (for Quality: Best - Full HD)", "type": None, "card": 4, "tested": True,
     "help": "Needed only for the Best quality: the second pass that makes 1080p.",
     "files": [{"url": HF + "Lightricks/LTX-Video/resolve/main/ltxv-spatial-upscaler-0.9.8.safetensors",
                "folder": "latent_upscale_models", "file": "ltxv-spatial-upscaler-0.9.8.safetensors", "size": 0.51 * GB}]},
]
TYPE_NAMES = {"ltxdist": "LTX-Video distilled checkpoint (8 steps)", "ltxdev": "LTX-Video dev checkpoint (30 steps)",
              "ltx13": "LTX-Video model only (GGUF or safetensors)", "vace": "Wan 2.1 VACE"}
SEARCH = ("checkpoints", "unet", "diffusion_models")  # where a video model file can live


def _path(folder, f):
    return os.path.join(MODELS, folder, os.path.basename(f))


def _have(x):
    p = _path(x["folder"], x["file"])
    return os.path.isfile(p) and os.path.getsize(p) >= x["size"] * 0.98


def free_gb():
    return shutil.disk_usage(MODELS).free / GB


# ---------- your own models ----------
def custom_models():
    try:
        return json.load(open(CUSTOM, encoding="utf-8"))
    except Exception:
        return []


def _save_custom(items):
    with open(CUSTOM, "w", encoding="utf-8") as fh:
        json.dump(items, fh, indent=1)


def find_file(f):
    for folder in SEARCH:
        p = _path(folder, f)
        if os.path.isfile(p):
            return folder, p
    return None, None


def model_files():
    """Every file in the three model folders, so the user can pick one to add."""
    out = []
    for folder in SEARCH:
        d = os.path.join(MODELS, folder)
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if f.endswith((".gguf", ".safetensors", ".ckpt", ".pt", ".pth", ".bin")):
                out.append({"file": f, "folder": folder, "gb": round(os.path.getsize(os.path.join(d, f)) / GB, 1)})
    return out


def _tensor_names(p):
    with open(p, "rb") as f:
        return _parse_header(f, p.lower().endswith(".gguf"))


def _parse_header(f, is_gguf):
    """Tensor names + a few header facts, read from the file header only (never loads the weights).
    f can be a real file or the first few MB fetched from Hugging Face."""
    if True:
        if is_gguf:
            if f.read(4) != b"GGUF":
                raise ValueError("bad")
            _, nt, nkv = struct.unpack("<IQQ", f.read(20))

            def s():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")
            size = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}

            def val(t):
                if t == 8:
                    return s()
                if t == 9:
                    it, n = struct.unpack("<IQ", f.read(12))
                    return [val(it) for _ in range(n)]
                return f.read(size[t])
            arch = ""
            for _ in range(nkv):
                k = s()
                t, = struct.unpack("<I", f.read(4))
                v = val(t)
                if k == "general.architecture":
                    arch = v
            names = {}
            for _ in range(nt):
                n = s()
                nd, = struct.unpack("<I", f.read(4))
                names[n] = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))[::-1]  # GGUF stores dims reversed
                f.read(12)
            return arch, names
        n, = struct.unpack("<Q", f.read(8))
        if n > 200_000_000:
            raise ValueError("bad")
        raw = f.read(n)
        if len(raw) < n:
            raise EOFError("header longer than what we read")
        h = json.loads(raw)
        h.pop("__metadata__", None)
        return "", {k: v.get("shape") for k, v in h.items()}


def check(f):
    """A file already in the ComfyUI model folders: what is it, and can this app run it?"""
    folder, p = find_file(f)
    if not p:
        return {"error": "File not found. Press + and pick the model file again."}
    return inspect(p, folder)


def inspect(p, folder=None):
    """What is this file, and can this app run it? -> {ok, type, kind, target, warn} or {error}.
    folder=None: a file the user picked anywhere on the PC; we decide which ComfyUI folder it belongs in."""
    if not p.lower().endswith((".gguf", ".safetensors")):
        return {"error": "This app reads only .gguf and .safetensors files. Old .ckpt/.pt/.bin files can hide code, so we don't open them."}
    try:
        arch, names = _tensor_names(p)
    except Exception:
        return {"error": "The file looks broken or only half downloaded (its header can't be read). Download it again."}
    return _judge(arch, names, os.path.basename(p), os.path.getsize(p), folder, local=True)


def _judge(arch, names, fname, nbytes, folder, local):
    """local=False: a file still on Hugging Face. Its helper files (text encoder, VAE) are then downloaded
    with it, so missing helpers are not an error."""
    p = fname
    keys = [k[len("model.diffusion_model."):] if k.startswith("model.diffusion_model.") else k for k in names]
    has = lambda s: any(s in k for k in keys)
    gb = nbytes / GB
    if has("lora_") or has(".lora") or has("lora_down") or has("lora_A"):
        return {"error": "This is a LoRA (a small add-on that changes a style or character), not a full video model. LoRAs aren't supported yet."}
    if has("encoder.block.") or has("shared.weight") or (has("encoder.") and has("decoder.") and not has("transformer_blocks")):
        return {"error": "This is a helper file (a text encoder or a VAE), not a video model. It's used by the models automatically."}
    if arch == "ltxv" or (has("adaln_single") and has("transformer_blocks")):
        if has("audio"):
            return {"error": "This is LTX-2 (video with sound). It's a newer design that this app can't run yet."}
        wide = next((v for k, v in zip(keys, names.values()) if k.endswith("transformer_blocks.0.attn1.to_q.weight")), [0])
        size = "13B" if max(wide or [0]) >= 4096 else "2B"
        f = os.path.basename(p)
        dev = "distill" not in f.lower()  # only files that say "distilled" use the 8-step settings
        import re
        old = re.search(r"v?0\.9(\.[0-4])?([^.\d]|$)", f.lower()) and not re.search(r"0\.9\.[5-9]", f.lower())
        if has("vae.") and folder in (None, "checkpoints"):
            t, target = ("ltxdev" if dev else "ltxdist"), "checkpoints"
            kind = "LTX-Video %s %s checkpoint" % (size, "dev" if dev else "distilled")
        elif has("vae."):
            return {"error": "This is a full LTX checkpoint (model + VAE in one file). Move it to ComfyUI/models/checkpoints, then press Refresh."}
        else:
            target = "unet" if p.lower().endswith(".gguf") else "diffusion_models"
            t = "ltx13dev" if dev else "ltx13"
            kind = "LTX-Video %s %s (model only, uses the VAE from the LTX 2B distilled file)" % (size, "dev" if dev else "distilled")
            if old:
                kind += ". Older LTX (0.9.0-0.9.4): it may look wrong with the newer VAE, a 0.9.5+ file is safer"
            if local and not _have(LTX_SMALL):
                return {"error": "This LTX file needs the LTX-Video 2B Distilled model too (it borrows its VAE). Download that one first from the list."}
        if local and not _have(T5):
            return {"error": "LTX needs its text encoder (t5xxl_fp8). Download any LTX model from the list once and it comes with it."}
        return _ok(t, kind, gb, folder and p.endswith(".gguf") and folder != "unet", target)
    if arch == "wan" or has("patch_embedding") and has("blocks.0.self_attn"):
        pe = next((v for k, v in zip(keys, names.values()) if k == "patch_embedding.weight"), None)
        if not has("vace_"):
            if pe and len(pe) > 1 and pe[1] == 48:
                return {"error": "This is Wan 2.2 TI2V 5B. We removed it: on our tests every render came out glitched."}
            return {"error": "This is a Wan model without VACE. The app runs only Wan 2.1 VACE (the one that keeps your picture as frame one). Look for a file with VACE in its name."}
        if local and not (_have(UMT5) and _have(WAN_VAE)):
            return {"error": "Wan needs its text encoder and VAE. Download 'Wan 2.1 VACE 1.3B' from the list once and they come with it."}
        size = "14B" if pe and pe[0] >= 5120 else "1.3B"
        return _ok("vace", "Wan 2.1 VACE %s" % size, gb, folder and p.endswith(".gguf") and folder != "unet",
                   "unet" if p.lower().endswith(".gguf") else "diffusion_models")
    return {"error": "This isn't a video model this app knows. It runs LTX-Video 0.9.x and Wan 2.1 VACE."}


def _ok(t, kind, gb, wrong_folder, target):
    if wrong_folder:
        return {"error": "GGUF files must be in ComfyUI/models/unet. Move it there, then press Refresh."}
    out = {"ok": True, "type": t, "kind": kind, "gb": round(gb, 1), "target": target}
    vram = vram_gb()
    if vram and gb > vram * 1.75:  # up to ~1.75x ran fine here (13B Q3, 6.5 GB, on a 4 GB card: ~7 min)
        out["warn"] = ("The file is %.1f GB and your graphics card has %.0f GB. It can still run (parts move to normal "
                       "memory) but much slower, and big sizes may fail with 'out of memory'." % (gb, vram))
    return out


def add(name, f):
    r = check(f)
    if not r.get("ok"):
        return r
    items = [c for c in custom_models() if c["file"] != f]
    items.append({"name": (name or f).strip()[:60], "type": r["type"], "file": os.path.basename(f), "kind": r["kind"]})
    _save_custom(items)
    return {"ok": True, "items": items}


# The normal Windows file window, opened by PowerShell. The Python that runs this server inside OpenDaemon
# (ComfyUI's embedded one) has no tkinter, so the "+" button failed with "No module named 'tkinter'".
# Windows won't give a background process the foreground, so the window opened behind the app (Alt+Tab to
# find it). Same fix as AI Developer's folder picker (extensions/browser/src/orchestrator/workspace.ts):
# a timer raises it with SetWindowPos once it exists, and here it is also centred on the screen with the mouse.
_PICK_PS = r"""
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class OdPickFront {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [StructLayout(LayoutKind.Sequential)] public struct RECT { public int Left, Top, Right, Bottom; }
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc cb, IntPtr l);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int cx, int cy, uint flags);
  [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr h);
  static bool centred = false;
  static IntPtr Mine() {
    uint me = (uint)System.Diagnostics.Process.GetCurrentProcess().Id;
    IntPtr found = IntPtr.Zero;
    EnumWindows(delegate(IntPtr h, IntPtr l) {
      uint pid; GetWindowThreadProcessId(h, out pid);
      if (pid == me && IsWindowVisible(h)) { found = h; return false; }
      return true;
    }, IntPtr.Zero);
    return found;
  }
  public static void Raise(bool focus, int areaX, int areaY, int areaW, int areaH) {
    IntPtr h = Mine();
    if (h == IntPtr.Zero) { return; }
    uint flags = 0x0001 | 0x0002 | 0x0040; // keep size, keep place, show
    int x = 0, y = 0;
    RECT r;
    if (!centred && GetWindowRect(h, out r)) {
      x = areaX + Math.Max(0, (areaW - (r.Right - r.Left)) / 2);
      y = areaY + Math.Max(0, (areaH - (r.Bottom - r.Top)) / 2);
      flags = 0x0001 | 0x0040; // move it once; after that the user may drag it
      centred = true;
    }
    SetWindowPos(h, new IntPtr(-1), x, y, 0, 0, flags);
    if (focus) { SetForegroundWindow(h); }
  }
}
"@
$area = [System.Windows.Forms.Screen]::FromPoint([System.Windows.Forms.Cursor]::Position).WorkingArea
$script:ticks = 0
$timer = New-Object System.Windows.Forms.Timer
$timer.Interval = 250
$timer.Add_Tick({
  $script:ticks = $script:ticks + 1
  [OdPickFront]::Raise($script:ticks -le 4, $area.X, $area.Y, $area.Width, $area.Height)
  if ($script:ticks -ge 16) { $timer.Stop() }
})
$timer.Start()
$d = New-Object System.Windows.Forms.OpenFileDialog
$d.Title = 'Pick a video model file'
$d.Filter = 'Video model files (*.gguf, *.safetensors)|*.gguf;*.safetensors'
$d.InitialDirectory = [Environment]::GetFolderPath('UserProfile') + '\Downloads'
$ok = $d.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK
$timer.Stop()
if ($ok) { [Console]::Out.Write($d.FileName) }
"""


def pick():
    """Opens the normal file window on this PC, showing only files this app can read. "" = cancelled."""
    if os.name == "nt":
        import base64
        enc = base64.b64encode(_PICK_PS.encode("utf-16-le")).decode()
        r = subprocess.run(["powershell", "-NoProfile", "-STA", "-NonInteractive", "-EncodedCommand", enc],
                           capture_output=True, timeout=1800, creationflags=0x08000000)  # no console window
        if r.returncode != 0:
            raise RuntimeError("the file window didn't open (" + r.stderr.decode("utf-8", "replace").strip()[:160] + ")")
        p = r.stdout.decode("utf-8", "replace").strip()
        return os.path.normpath(p) if p else ""
    import tkinter
    from tkinter import filedialog
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # in front of the browser, not hidden behind it
    try:
        p = filedialog.askopenfilename(parent=root, title="Pick a video model file",
                                       initialdir=os.path.join(os.path.expanduser("~"), "Downloads"),
                                       filetypes=[("Video model files (.gguf, .safetensors)", "*.gguf *.safetensors")])
    finally:
        root.destroy()
    return os.path.normpath(p) if p else ""


def add_path(p, name=None):
    """Adds a model file from anywhere on the PC, without moving or copying it: ComfyUI only loads files
    from its own folders, so we put a link there (a second name for the same file, takes no space)."""
    if not p or not os.path.isfile(p):
        return {"error": "File not found."}
    r = inspect(p)
    if not r.get("ok"):
        return r
    d = os.path.join(MODELS, r["target"])
    base, ext = os.path.splitext(os.path.basename(p))
    link, n = os.path.join(d, base + ext), 1
    while os.path.exists(link) and not os.path.samefile(link, p):
        n += 1
        link = os.path.join(d, "%s (%d)%s" % (base, n, ext))
    made = False
    if not os.path.exists(link):
        try:
            os.link(p, link)  # hard link: same drive, no admin rights needed
        except OSError:
            try:
                os.symlink(p, link)  # other drive: needs Windows Developer Mode
            except OSError:
                return {"error": "The model is on drive %s and ComfyUI's folders are on drive %s. The app can't point at a file "
                                 "on another drive without copying it. Move the file to drive %s (any folder, Downloads is fine) "
                                 "and press + again." % (os.path.splitdrive(p)[0], os.path.splitdrive(MODELS)[0], os.path.splitdrive(MODELS)[0])}
        made = True
    f = os.path.basename(link)
    items = [c for c in custom_models() if c["file"] != f]
    items.append({"name": (name or base).strip()[:60], "type": r["type"], "file": f, "kind": r["kind"],
                  "path": p, "linked": made, "folder": r["target"]})
    _save_custom(items)
    return dict(r, file=f, items=items)


def remove(f):
    items = custom_models()
    c = next((x for x in items if x["file"] == f), None)
    if c and c.get("linked"):
        link = _path(c.get("folder") or "unet", f)
        # delete only the link we made, and only while it still is the user's file (their original stays)
        if os.path.isfile(link) and (not os.path.isfile(c["path"]) or os.path.samefile(link, c["path"])):
            os.remove(link)
    _save_custom([x for x in items if x["file"] != f])
    return {"ok": True}


# ---------- one-click downloads ----------
DL = {"id": None}  # the one running download: id, name, done, total, error, cancel
_VRAM = {}


def vram_gb():
    if "v" not in _VRAM:
        try:
            with urllib.request.urlopen(paths.URL + "/system_stats", timeout=5) as r:
                _VRAM["v"] = json.loads(r.read())["devices"][0]["vram_total"] / 2**30
        except Exception:
            return None
    return _VRAM["v"]


def listed_models():
    """The extra models for the Model list: the ones you added, plus every catalog model whose files are on
    this PC. A finished download used to be written only into the list of the data folder it ran from, so a
    model downloaded by the dev copy (11-18, LTX 13B Q4) never showed in OpenDaemon. Files on disk are the truth.
    Downloaded ones carry "downloaded": True (removed from the Models list, not with the Remove button)."""
    items = custom_models()
    have = {c["file"] for c in items}
    for m in CATALOG:
        f = m["files"][0]["file"]
        if m.get("builtin") or not m.get("type") or f in have or any(not _have(x) for x in m["files"]):
            continue
        items.append({"name": m["name"], "type": m["type"], "file": f, "kind": m["name"], "downloaded": True,
                      "card": m.get("card")})
    return items


def catalog():
    custom_files = {c["file"] for c in listed_models()}
    vram = vram_gb()
    fits = [m for m in CATALOG if m.get("rank") and vram and m["card"] <= vram + 0.5]
    best = max(fits, key=lambda m: m["rank"])["id"] if fits else None
    out = []
    for m in CATALOG:
        need = [x for x in m["files"] if not _have(x)]
        out.append({k: m.get(k) for k in ("id", "name", "help", "card", "tested", "builtin", "type")} | {
            "installed": not need, "need_gb": round(sum(x["size"] for x in need) / GB, 1),
            "main": m["files"][0]["file"], "listed": m.get("builtin") is not None or m["files"][0]["file"] in custom_files,
            "best": m["id"] == best})
    out.sort(key=lambda c: not c["best"])  # the best one first, the rest keep their order below
    return out


def start_download(mid):
    m = next((x for x in CATALOG if x["id"] == mid), None)
    if not m:
        return {"error": "Unknown model."}
    return _start(m)


def _start(m):
    mid = m["id"]
    if DL.get("id") and not DL.get("finished"):
        return {"error": "Another download is running. Wait for it to finish, or cancel it."}
    need = [x for x in m["files"] if not _have(x)]
    total = sum(x["size"] for x in need)
    if total + 1 * GB > free_gb() * GB:
        return {"error": "Not enough disk space: this needs %.1f GB and only %.1f GB is free (we keep 1 GB spare). Free some space first." % (total / GB, free_gb())}
    DL.clear()
    DL.update(id=mid, name=m["name"], done=0, total=total, error=None, cancel=False, finished=False, started=time.time())
    threading.Thread(target=_download, args=(m, need), daemon=True).start()
    return {"ok": True}


def _download(m, need):
    try:
        base = 0
        for x in need:
            dst = _path(x["folder"], x["file"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            part = dst + ".part"
            have = os.path.getsize(part) if os.path.isfile(part) else 0  # resume a stopped download
            req = urllib.request.Request(x["url"], headers={"Range": "bytes=%d-" % have} if have else {})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have and r.status == 206 else "wb") as fh:
                got = have if r.status == 206 else 0
                while True:
                    if DL.get("cancel"):
                        DL.update(error="Cancelled. Start it again to continue where it stopped.", finished=True)
                        return
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    got += len(b)
                    DL["done"] = base + got
            if os.path.getsize(part) < x["size"] * 0.98:
                raise RuntimeError("the download stopped early. Start it again to continue.")
            os.replace(part, dst)
            base += x["size"]
        if not m.get("builtin") and m.get("type"):
            items = [c for c in custom_models() if c["file"] != m["files"][0]["file"]]
            items.append({"name": m["name"], "type": m["type"], "file": m["files"][0]["file"], "kind": m.get("kind") or m["name"],
                          "source": m.get("source")})
            _save_custom(items)
        DL.update(done=DL["total"], finished=True)
    except Exception as e:
        msg = str(e)
        if "urlopen error" in msg or "timed out" in msg or "getaddrinfo" in msg:
            msg = "No connection to Hugging Face (check the internet). Start it again to continue where it stopped."
        elif "No space" in msg or "Errno 28" in msg:
            msg = "The disk is full. Free some space, then start it again."
        elif "HTTP Error 404" in msg:
            msg = "The file is no longer at that address on Hugging Face (404). Tell us so we can fix the link."
        DL.update(error="Download failed: " + msg, finished=True)


# ---------- advanced: search Hugging Face, always capped to what this card can run ----------
HF_API = HF + "api/models"
SKIP = ("lora", "vae", "text_encoder", "text-encoder", "umt5", "t5xxl", "clip", "mmproj", "upscaler", "encoder", "controlnet")
OUR_URLS = {m["files"][0]["url"] for m in CATALOG}  # Advanced marks these "In our list"
HF_FILES = {}  # (repo, file) -> bytes; only files a search showed can be downloaded (keeps the size cap real)


def cap_gb():
    """Biggest file we show for this card: ~1.75x the graphics memory is what still ran here (offloading to RAM)."""
    return round((vram_gb() or 4) * 1.75, 1)


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "video-pipeline"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def hf_search(q=""):
    import re
    import urllib.parse
    from concurrent.futures import ThreadPoolExecutor
    cap = cap_gb()
    queries = [q.strip()] if q.strip() else ["LTXV", "LTX-Video", "VACE"]  # no text: the families this app runs
    repos = []
    for qq in queries:
        for m in _get_json(HF_API + "?" + urllib.parse.urlencode({"search": qq, "limit": 30, "sort": "downloads", "direction": -1})):
            if m["id"] not in repos:
                repos.append(m["id"])

    def info(rid):
        try:
            return _get_json(HF_API + "/" + urllib.parse.quote(rid, safe="/") + "?blobs=true")
        except Exception:
            return None
    with ThreadPoolExecutor(8) as ex:
        infos = list(ex.map(info, repos[:40]))
    out, too_big = [], 0
    for d in infos:
        if not d or d.get("private") or d.get("gated") or d.get("disabled"):
            continue
        files = []
        for s in d.get("siblings", []):
            n, size = s["rfilename"], s.get("size") or 0
            low = n.lower()
            if not low.endswith((".gguf", ".safetensors")) or size < 1 * GB:
                continue
            if any(k in os.path.basename(low) for k in SKIP) or re.search(r"-\d{5}-of-\d{5}", low):
                continue  # helpers, add-ons, and GGUF split into parts (ComfyUI can't load those)
            if size > cap * GB:
                too_big += 1
                continue
            HF_FILES[(d["id"], n)] = size
            ours = (HF + d["id"] + "/resolve/main/" + n) in OUR_URLS
            files.append({"file": n, "gb": round(size / GB, 1), "ours": ours})
        if files:
            files.sort(key=lambda f: f["gb"])
            out.append({"repo": d["id"], "likes": d.get("likes", 0), "downloads": d.get("downloads", 0),
                        "updated": (d.get("lastModified") or "")[:10], "files": files[:15]})
    out.sort(key=lambda r: -r["downloads"])
    return {"cap": cap, "vram": round(vram_gb() or 0, 1), "repos": out, "hidden": too_big}


def _hf_url(repo, f):
    import urllib.parse
    return HF + repo + "/resolve/main/" + urllib.parse.quote(f, safe="/")


def hf_check(repo, f):
    """Reads only the first MB of the file on Hugging Face (its header) to see if this app can run it,
    so nobody waits for a 10 GB download to find out."""
    import io
    size = HF_FILES.get((repo, f))
    if not size:
        return {"error": "Search again first: only files the search showed (and that fit your card) can be downloaded."}
    names = None
    for n in (4 << 20, 48 << 20):
        req = urllib.request.Request(_hf_url(repo, f), headers={"Range": "bytes=0-%d" % (n - 1), "User-Agent": "video-pipeline"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read(n)
        except Exception as e:
            return {"error": "Couldn't read the file on Hugging Face (%s). It may need a sign-in or be removed." % str(e)[:120]}
        try:
            arch, names = _parse_header(io.BytesIO(data), f.lower().endswith(".gguf"))
            break
        except Exception:
            continue
    if names is None:
        return {"error": "The file's header can't be read: it looks broken or isn't a normal model file. Don't download it."}
    r = _judge(arch, names, os.path.basename(f), size, None, local=False)
    if r.get("ok"):
        r["size"] = size
    return r


def hf_download(repo, f):
    r = hf_check(repo, f)
    if not r.get("ok"):
        return r
    main = {"url": _hf_url(repo, f), "folder": r["target"], "file": os.path.basename(f), "size": r["size"]}
    deps = {"ltx13": [LTX_SMALL, T5], "ltx13dev": [LTX_SMALL, T5], "ltxdist": [T5], "ltxdev": [T5], "vace": [UMT5, WAN_VAE]}[r["type"]]
    name = os.path.splitext(os.path.basename(f))[0]
    return dict(_start({"id": "hf:" + repo + "/" + f, "name": name, "type": r["type"], "kind": r["kind"],
                        "source": HF + repo, "files": [main] + deps}), kind=r["kind"])


def status():
    return {"catalog": catalog(), "custom": listed_models(), "files": model_files(), "download": DL if DL.get("id") else None,
            "vram": round(vram_gb() or 0, 1), "free": round(free_gb(), 1)}


def open_folder(folder):
    if folder in SEARCH:
        os.startfile(os.path.join(MODELS, folder))
    return {"ok": True}
