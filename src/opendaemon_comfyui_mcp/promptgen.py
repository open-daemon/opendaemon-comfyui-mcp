"""Turns the user's idea + the start picture into a clean LTX video prompt, using a vision model.

For now it borrows a key from the OpenDaemon config on this PC (read at call time, never stored here).
Later this becomes a call to the app's own model.
"""
import base64
import io
import json
import os
import urllib.request

CONFIG = os.path.expanduser(r"~\.OpenDaemon\user-data\current\demo-home\.opendeamon\opendeamon.json")
NVIDIA_MODEL = "nvidia/nemotron-3.5-super-vl-120b-a12b"
GEMINI_MODEL = "gemini-3.5-flash"
TEXT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"  # used only when no vision model answers

RULES = """You write prompts for LTX-Video, an image-to-video model. It gets the picture as the first frame plus your prompt.
Rules that work for this model (tested):
- Do NOT describe the picture again (looks, colours, background). The model already sees it. Re-describing makes it worse.
- Describe only the MOTION: who moves, how, in what direction, how fast. One short action per character.
- Plain present tense, simple words, 1 to 3 short sentences, under 45 words in total.
- Refer to characters simply and so they can be told apart ("the boy", "the red mascot").
- Clips are only 2 to 5 seconds: keep the action small and finishable in that time.
- If a character should not talk, say "with his mouth closed" (talking mouths come out as mumbling).
- Prefer actions without detailed hand work; hands come out wrong.
- End with exactly one camera sentence: "Static camera." unless the user asked for a camera move.
Answer with the prompt only: no quotes, no title, no explanation."""


ENV_KEYS = {"google": ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "nvidia": ("NVIDIA_API_KEY",)}
CONFIGS = [os.environ.get("OPENDAEMON_CONFIG", ""), CONFIG, os.path.expanduser(os.path.join("~", ".opendeamon", "opendeamon.json"))]


def _keys(provider):
    """API keys for the vision helpers: environment first (GEMINI_API_KEY, NVIDIA_API_KEY; several comma-separated),
    then the OpenDaemon config on this PC. Read at call time, never stored. No key = an empty list."""
    for name in ENV_KEYS.get(provider, ()):
        v = os.environ.get(name, "")
        if v.strip():
            return [x.strip() for x in v.split(",") if x.strip()]
    for path in CONFIGS:
        try:
            d = json.load(open(path, encoding="utf-8"))
            k = str(d["models"]["providers"][provider].get("apiKey", ""))
        except Exception:
            continue
        ks = [x.strip() for x in k.split(",") if x.strip()]
        if ks:
            return ks
    return []


def no_key_hint():
    return ("No vision model key found. Set GEMINI_API_KEY (free at aistudio.google.com) or NVIDIA_API_KEY "
            "(free at build.nvidia.com) in the MCP server's environment.")


def _jpeg(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((1024, 1024))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=88)
    return base64.b64encode(b.getvalue()).decode()


def _post(url, body, headers, timeout=40):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _clean(t):
    t = (t or "").strip()
    if "</think>" in t:
        t = t.split("</think>")[-1].strip()
    return t.strip().strip('"').strip()


NEG_RULES = """You write NEGATIVE prompts for an image-to-video model (LTX-Video or Wan). It gets the picture as the first frame,
a normal prompt saying what should happen, and your negative prompt: a list of things the video must NOT show.
The model pushes the video AWAY from every word you write, so:
- Write only short, concrete things to avoid, separated by commas. No sentences, no "no", no "don't" (write "blurry", not "not blurry").
- Look at the picture and the planned motion and name what is likely to go wrong in THIS clip: e.g. extra arms or hands,
  extra fingers, melting or warping faces, a character changing shape or colour, mouths moving or talking when they should not,
  flicker, blur, jitter, text or watermarks appearing, objects popping in or out, camera shake.
- Never list anything the prompt WANTS (if the character should spin, do not write "spinning").
- Always include the basics: worst quality, blurry, jittery, distorted, deformed, watermark, text.
- 15 to 35 items, under 60 words.
Answer with the comma-separated list only: no quotes, no title, no explanation."""


def generate(image_path, idea, kind="prompt", negative=""):
    """kind="prompt": write the video prompt from the user's idea. kind="negative": write a negative prompt for it."""
    img = _jpeg(image_path)
    if kind == "negative":
        rules = NEG_RULES
        user = (f"The prompt for this clip (what SHOULD happen):\n{idea.strip()}\n\n"
                + (f"The user's current negative prompt (keep what still fits):\n{negative.strip()}\n\n" if negative.strip() else "")
                + "Write the negative prompt.")
    else:
        rules = RULES
        user = f"The user's idea for this clip:\n{idea.strip()}\n\nWrite the LTX prompt."
    r = _generate(img, user, rules)
    if kind == "negative":
        r["prompt"] = _drop_wanted(r["prompt"], idea)
    return r


def _drop_wanted(neg, prompt):
    """Never avoid what the prompt asks for: drop list items that share a word stem with the prompt's action words."""
    import re
    skip = {"static", "camera", "with", "the", "and", "his", "her", "its", "mouth", "closed", "slowly", "once"}
    stems = {w[:4] for w in re.findall(r"[a-z]+", prompt.lower()) if len(w) > 3 and w not in skip}
    keep = [x.strip() for x in neg.split(",") if x.strip()
            and not any(w[:4] in stems for w in re.findall(r"[a-z]+", x.lower()) if len(w) > 3)]
    return ", ".join(keep)


def _generate(img, user, rules):
    errors = []
    # 1) Gemini (fast, good at pictures)
    for key in _keys("google")[:1]:
        try:
            r = _post(f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                      {"systemInstruction": {"parts": [{"text": rules}]},
                       "contents": [{"role": "user", "parts": [{"inline_data": {"mime_type": "image/jpeg", "data": img}}, {"text": user}]}],
                       "generationConfig": {"temperature": 0.6, "maxOutputTokens": 2048}},
                      {"x-goog-api-key": key})
            t = _clean("".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"]))
            if t:
                return {"prompt": t, "model": GEMINI_MODEL}
        except Exception as e:
            errors.append(f"gemini: {str(e)[:120]}")
    # 2) NVIDIA vision model, trying a few keys
    for key in _keys("nvidia")[:2]:
        try:
            r = _post("https://integrate.api.nvidia.com/v1/chat/completions",
                      {"model": NVIDIA_MODEL, "temperature": 0.6, "max_tokens": 1500, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": rules},
                                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + img}},
                                                                 {"type": "text", "text": user}]}]},
                      {"Authorization": "Bearer " + key})
            t = _clean(r["choices"][0]["message"].get("content"))
            if t:
                return {"prompt": t, "model": NVIDIA_MODEL}
        except Exception as e:
            errors.append(f"nvidia: {str(e)[:120]}")
    # 3) Text-only fallback: the model can't see pictures, so it works from the user's words alone
    text_user = user + "\n\n(You cannot see the picture. Use only what the user wrote; do not invent how things look.)"
    for key in _keys("nvidia")[:2]:
        try:
            r = _post("https://integrate.api.nvidia.com/v1/chat/completions",
                      {"model": TEXT_MODEL, "temperature": 0.6, "max_tokens": 4000, "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": rules}, {"role": "user", "content": text_user}]},
                      {"Authorization": "Bearer " + key})
            t = _clean(r["choices"][0]["message"].get("content"))
            if t:
                return {"prompt": t, "model": TEXT_MODEL + " (text only, did not see the picture)"}
        except Exception as e:
            errors.append(f"text: {str(e)[:120]}")
    if not errors:
        raise RuntimeError(no_key_hint())
    raise RuntimeError("No model answered: " + " | ".join(errors))


if __name__ == "__main__":
    import sys
    print(generate(sys.argv[1], sys.argv[2]))
