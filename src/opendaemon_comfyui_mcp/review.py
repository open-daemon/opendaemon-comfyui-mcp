"""Self-check for finished videos: pull frames, let a vision model judge the clip, and pick a fix.

The fix rules come from our own tests on a 4 GB card (see the memory notes): drift = less movement,
no action = more movement, melting at the end = shorter clip, and always a new seed.
"""
import glob
import json
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
import paths
OUT = paths.OUT
FRAMES = os.path.join(OUT, "frames")
PROBLEMS = {
    "drift": "the character or picture changes into something else (face, shape, colours)",
    "wrong_subject": "a new or wrong subject appears (e.g. a dog instead of the cat)",
    "no_action": "the asked action does not happen, or barely anything moves",
    "too_much_motion": "wild, chaotic movement",
    "hands": "extra, missing or broken hands, arms or fingers",
    "melting": "parts melt, warp or smear",
    "end_melts": "the clip is fine at the start but falls apart near the end",
    "talking": "a mouth moves or talks when it should not",
    "blurry": "blurry, flickering or low detail",
}


def ffmpeg():
    c = paths.ffmpeg_candidates()
    return c[0] if c else "ffmpeg"


def frames(video, comfy_output, n=3):
    """n frames spread over the clip (first, middle, last for n=3), as JPG files. Uses the render's own PNG
    frames when ComfyUI still has them, else cuts them from the mp4."""
    from PIL import Image
    name = os.path.splitext(os.path.basename(video))[0]
    n = max(1, min(int(n), 9))
    os.makedirs(FRAMES, exist_ok=True)
    pngs = sorted(glob.glob(os.path.join(comfy_output, name, "f_*.png")))
    out = []
    if pngs:
        picks = [pngs[round(i * (len(pngs) - 1) / max(1, n - 1))] for i in range(n)] if n > 1 else [pngs[len(pngs) // 2]]
        for i, p in enumerate(picks):
            dst = os.path.join(FRAMES, "%s_%d.jpg" % (name, i))
            Image.open(p).convert("RGB").save(dst, "JPEG", quality=85)
            out.append(dst)
        return out
    mp4 = os.path.join(OUT, name + ".mp4")
    s = json.load(open(os.path.join(OUT, name + ".json"))) if os.path.exists(os.path.join(OUT, name + ".json")) else {}
    secs = max(0.2, (int(s.get("frames", 49)) - 1) / int(s.get("fps", 24)))
    for i in range(n):
        t = secs / 2 if n == 1 else (secs - 0.05) * i / (n - 1)
        dst = os.path.join(FRAMES, "%s_%d.jpg" % (name, i))
        subprocess.run([ffmpeg(), "-v", "error", "-y", "-ss", "%.2f" % t, "-i", mp4, "-frames:v", "1", "-q:v", "3", dst], check=True)
        out.append(dst)
    return out


def sheet(paths, dst, height=288):
    """Frames side by side in one picture (cheaper for a model to look at than separate pictures)."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    ims = [im.resize((round(im.width * height / im.height), height)) for im in ims]
    gap = 8
    out = Image.new("RGB", (sum(im.width for im in ims) + gap * (len(ims) - 1), height), (20, 20, 20))
    x = 0
    for im in ims:
        out.paste(im, (x, 0))
        x += im.width + gap
    out.save(dst, "JPEG", quality=85)
    return dst


JUDGE_RULES = """You check short AI videos made from a picture (image-to-video). You get the START PICTURE, then a strip of
frames from the video (start, middle, end, left to right), and the prompt that says what should happen.
Judge honestly; these models often fail. Answer with JSON only:
{"score": 0-10, "action_happened": true/false, "same_character": true/false, "problems": [...], "note": "one short sentence"}
problems: only from this list, only if you clearly see them: %s.
score: 8-10 = usable as is, 5-7 = small flaws, 0-4 = broken. No problems and score >= 7 means the clip is fine."""


TEXT_ONLY_NOTE = """
This clip was made from the prompt alone (text to video): there is NO start picture. Judge the frames against the
prompt; "same_character" means the subject stays the same from start to end."""


def judge(start_image, strip, prompt):
    """Asks a vision model (Gemini, then NVIDIA; keys from the environment or OpenDaemon's config) to judge the clip.
    start_image may be missing (a clip made from the prompt alone): then only the frames and the prompt are sent."""
    import promptgen
    pic = bool(start_image) and os.path.isfile(start_image)
    rules = JUDGE_RULES % ", ".join('"%s" (%s)' % kv for kv in PROBLEMS.items()) + ("" if pic else TEXT_ONLY_NOTE)
    user = "The prompt: " + prompt
    a, b = (promptgen._jpeg(start_image) if pic else None), promptgen._jpeg(strip)
    errors = []
    for key in promptgen._keys("google")[:1]:
        try:
            r = promptgen._post(f"https://generativelanguage.googleapis.com/v1beta/models/{promptgen.GEMINI_MODEL}:generateContent",
                                {"systemInstruction": {"parts": [{"text": rules}]},
                                 "contents": [{"role": "user", "parts": ([
                                     {"text": "START PICTURE:"}, {"inline_data": {"mime_type": "image/jpeg", "data": a}}] if pic else []) + [
                                     {"text": "VIDEO FRAMES (start, middle, end):"}, {"inline_data": {"mime_type": "image/jpeg", "data": b}},
                                     {"text": user}]}],
                                 "generationConfig": {"temperature": 0.2, "maxOutputTokens": 2048, "responseMimeType": "application/json"}},
                                {"x-goog-api-key": key}, timeout=60)
            return _verdict("".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"]), promptgen.GEMINI_MODEL)
        except Exception as e:
            errors.append("gemini: " + str(e)[:150])
    for key in promptgen._keys("nvidia")[:2]:
        try:
            r = promptgen._post("https://integrate.api.nvidia.com/v1/chat/completions",
                                {"model": promptgen.NVIDIA_MODEL, "temperature": 0.2, "max_tokens": 1500,
                                 "chat_template_kwargs": {"enable_thinking": False},
                                 "messages": [{"role": "system", "content": rules}, {"role": "user", "content": ([
                                     {"type": "text", "text": "START PICTURE:"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + a}}] if pic else []) + [
                                     {"type": "text", "text": "VIDEO FRAMES (start, middle, end):"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}},
                                     {"type": "text", "text": user}]}]},
                                {"Authorization": "Bearer " + key}, timeout=90)
            return _verdict(r["choices"][0]["message"].get("content"), promptgen.NVIDIA_MODEL)
        except Exception as e:
            errors.append("nvidia: " + str(e)[:150])
    if not errors:
        raise RuntimeError(promptgen.no_key_hint())
    raise RuntimeError("No vision model answered: " + " | ".join(errors))


def _verdict(text, model):
    text = promptgen_clean(text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("judge gave no JSON: " + text[:200])
    v = json.loads(m.group(0))
    probs = [p for p in v.get("problems") or [] if p in PROBLEMS]
    score = max(0, min(10, int(round(float(v.get("score", 0))))))
    return {"score": score, "ok": score >= 7 and not probs, "problems": probs, "action_happened": bool(v.get("action_happened")),
            "same_character": bool(v.get("same_character")), "note": str(v.get("note", ""))[:300], "judge": model}


def promptgen_clean(t):
    t = (t or "").strip()
    return t.split("</think>")[-1].strip() if "</think>" in t else t


AVOID_FOR = {"hands": "extra arms, extra hands, extra fingers, deformed hands", "melting": "melting, warping, smeared, distorted",
             "talking": "talking, moving mouth, lip movement", "blurry": "blurry, flicker, low detail",
             "wrong_subject": "new characters, different animal, extra creatures", "drift": "changing face, different character"}


def fix(s, problems):
    """New settings for another try, and why, in plain words. Always a new seed."""
    s, why = dict(s), []
    comp = int(float(s.get("comp", 35)))
    if {"drift", "wrong_subject", "too_much_motion"} & set(problems):
        s["comp"] = max(20, comp - 10)
        s["cfps"] = max(int(float(s.get("cfps", 24))), 30)
        why.append("less movement and stay closer to the picture (movement %d -> %s)" % (comp, s["comp"]))
    elif "no_action" in problems:
        s["comp"] = min(60, comp + 10)
        why.append("more movement so the action happens (movement %d -> %s)" % (comp, s["comp"]))
    if {"end_melts"} & set(problems) and int(s.get("frames", 73)) > 49:
        old = int(s["frames"])
        s["frames"] = max(49, old - 24) if s.get("family") != "vace" else max(33, old - 16)
        why.append("shorter clip so the end holds together (%d -> %d frames)" % (old, s["frames"]))
    neg = s.get("negative") or ""
    if neg or s.get("family") == "vace" or int(float(s.get("steps", 8))) >= 20:  # only these models read the avoid list
        add = [AVOID_FOR[p] for p in problems if p in AVOID_FOR and AVOID_FOR[p] not in neg]
        if add:
            s["negative"] = (neg + ", " if neg else "") + ", ".join(add)
            why.append("added to the avoid list: " + "; ".join(add))
    if "talking" in problems and "mouth closed" not in s.get("prompt", ""):
        s["prompt"] = re.sub(r"\s*(Static camera|Slow camera[^.]*)\.\s*$", "", s["prompt"]).strip() + " Mouth closed. " + \
            (re.search(r"(Static camera|Slow camera[^.]*)\.\s*$", s["prompt"]).group(0).strip() if re.search(r"(Static camera|Slow camera[^.]*)\.\s*$", s["prompt"]) else "")
        why.append("asked for a closed mouth in the prompt")
    s["seed"] = -1
    why.append("new seed")
    s.pop("easy", None)
    s["easy"] = "Fix try: " + ", ".join(problems or ["new seed"])
    return s, why
