"""Render graphs for the non-LTX-2B video models (GGUF, loaded with the ComfyUI-GGUF add-on).

family = "vace"   Wan 2.1 VACE 1.3B: picture held as frame 0 of a control video, rest generated, 16 fps
                  (no picture = text to video: no control video, every frame generated)
family = "ltx13"  LTX 0.9.8 13B distilled: same graph as the 2B, model swapped (VAE borrowed from the 2B checkpoint)
"""

WAN_TE = "umt5-xxl-encoder-Q4_K_M.gguf"
GRAY, BLACK, WHITE = 0x7F7F7F, 0x000000, 0xFFFFFF


def _common_wan(s, unet, vae):
    w, h, n = int(s["width"]), int(s["height"]), int(s["frames"])
    return w, h, n, {
        "1": ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}} if unet.endswith(".gguf") else
              {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}}),
        "2": {"class_type": "CLIPLoaderGGUF", "inputs": {"clip_name": s.get("wan_te") or WAN_TE, "type": "wan"}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": s["prompt"], "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": s.get("negative") or "", "clip": ["2", 0]}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "8": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["1", 0], "shift": float(s.get("shift") or 8)}},
    }


def _picture(s, wf, w, h):
    wf["6"] = {"class_type": "LoadImage", "inputs": {"image": s["image"]}}
    wf["6s"] = {"class_type": "ImageScale", "inputs": {"image": ["6", 0], "upscale_method": "lanczos",
                                                       "width": w, "height": h, "crop": "center"}}


def _sample_decode(s, wf, pos, neg, latent, vae_ref, outdir):
    wf["9"] = {"class_type": "KSampler", "inputs": {
        "model": ["8", 0], "seed": int(s["seed"]), "steps": int(s["steps"]), "cfg": float(s["cfg"]),
        "sampler_name": s.get("sampler") or "uni_pc", "scheduler": s.get("scheduler") or "simple",
        "positive": pos, "negative": neg, "latent_image": latent, "denoise": 1.0}}
    return wf


# Wan 2.2 TI2V 5B (GGUF Q4) was removed 2026-11-18: every render came out glitched on this card, files deleted.


def vace(s, outdir):
    w, h, n, wf = _common_wan(s, s["unet"], "wan_2.1_vae.safetensors")
    if not s.get("image"):
        wf["7"] = {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "vae": ["5", 0], "width": w, "height": h, "length": n,
            "batch_size": 1, "strength": float(s.get("strength") or 1)}}
        return _decode_vace(s, wf, outdir)
    _picture(s, wf, w, h)
    # control video: frame 0 = the picture (kept), frames 1.. = grey (to be generated); mask: 0 keep, 1 generate
    wf.update({
        "20": {"class_type": "EmptyImage", "inputs": {"width": w, "height": h, "batch_size": n - 1, "color": GRAY}},
        "21": {"class_type": "ImageBatch", "inputs": {"image1": ["6s", 0], "image2": ["20", 0]}},
        "22": {"class_type": "EmptyImage", "inputs": {"width": w, "height": h, "batch_size": 1, "color": BLACK}},
        "23": {"class_type": "EmptyImage", "inputs": {"width": w, "height": h, "batch_size": n - 1, "color": WHITE}},
        "24": {"class_type": "ImageBatch", "inputs": {"image1": ["22", 0], "image2": ["23", 0]}},
        "25": {"class_type": "ImageToMask", "inputs": {"image": ["24", 0], "channel": "red"}},
        "7": {"class_type": "WanVaceToVideo", "inputs": {
            "positive": ["3", 0], "negative": ["4", 0], "vae": ["5", 0], "width": w, "height": h, "length": n,
            "batch_size": 1, "strength": float(s.get("strength") or 1), "control_video": ["21", 0],
            "control_masks": ["25", 0], "reference_image": ["6s", 0]}},
    })
    return _decode_vace(s, wf, outdir)


def _decode_vace(s, wf, outdir):
    _sample_decode(s, wf, ["7", 0], ["7", 1], ["7", 2], ["5", 0], outdir)
    wf["12"] = {"class_type": "TrimVideoLatent", "inputs": {"samples": ["9", 0], "trim_amount": ["7", 3]}}
    wf["10"] = {"class_type": "VAEDecodeTiled", "inputs": {"samples": ["12", 0], "vae": ["5", 0],
                "tile_size": int(s.get("tile") or 256), "overlap": 64, "temporal_size": 32, "temporal_overlap": 8}}
    wf["11"] = {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": outdir + "/f"}}
    return wf


def ltx13(wf, s):
    """Takes the normal LTX graph and swaps the model for the 13B GGUF (VAE stays from the 2B checkpoint)."""
    wf["1m"] = ({"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": s["unet"]}} if s["unet"].endswith(".gguf") else
                {"class_type": "UNETLoader", "inputs": {"unet_name": s["unet"], "weight_dtype": "default"}})
    for node in wf.values():
        for k, v in node["inputs"].items():
            if v == ["1", 0]:
                node["inputs"][k] = ["1m", 0]
    return wf
