# OpenDaemon ComfyUI MCP

Make AI videos on your own computer, by asking your AI assistant.

This is an MCP server. It lets any AI app that speaks MCP (Claude Desktop, Claude Code, Cursor, OpenDaemon and
others) drive [ComfyUI](https://github.com/comfyanonymous/ComfyUI) on your own graphics card. The assistant can:

- **render** clips from a picture and an idea (image to video), or from the prompt alone (text to video),
  with settings picked for your card;
- **check its own work**: it looks at frames of each clip, scores it and re-renders what went wrong (drift,
  melting, no motion...), up to a few tries;
- **mass-create**: batches and variations wait in one queue, and your own renders always go first;
- **edit like a video editor**: join, trim, speed, extend a clip, captions, voice-over, music, title and end
  cards, logo, reformat to 9:16 and export for TikTok, Reels, Shorts, YouTube...;
- **run your own ComfyUI workflows**, check them before a slow run and install missing node packs (it asks you first).

Everything runs locally and costs nothing. Tested on a GTX 1050 Ti with 4 GB.

## How it works

```
AI app (MCP client) --MCP--> this server --HTTP--> pipeline server :8190 --HTTP--> ComfyUI :8188 --> your GPU
                                                     (queue, editing, web page)
```

The AI app is the client. This server gives it tools (67 of them). When the model calls a tool, the server passes
the job to a small pipeline server, which keeps one queue, talks to ComfyUI over its API and does the editing with
ffmpeg. The pipeline server also serves a web page (http://127.0.0.1:8190) where you can watch the queue and
render by hand. Renders started by the AI show up there too. If it isn't running, the MCP server starts it.

This project talks to ComfyUI only over its API and never bundles it.

## What you need

- **Windows** with an NVIDIA graphics card, 4 GB or more. Other systems may work but aren't tested yet.
- **ComfyUI**, for example the [portable build](https://github.com/comfyanonymous/ComfyUI/releases), plus the
  [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) node pack for the GGUF models.
- **Python 3.10 or newer**.
- A video model. Ask the assistant: "list the video models" and "download the one that fits my card". Models come
  straight from their makers' Hugging Face pages and keep their own licences. Read them before commercial use.

## Install

```bash
pip install git+https://github.com/open-daemon/opendaemon-comfyui-mcp
```

Or from a downloaded copy of this repo: `pip install .`

That installs two commands: `opendaemon-comfyui-mcp` (the MCP server) and `opendaemon-comfyui-web` (only the
web page).

## Connect your AI app

Set `COMFYUI_DIR` to the folder that holds ComfyUI's `main.py`, unless ComfyUI is in a usual place.

**Claude Desktop**: in `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "opendaemon-comfyui-mcp",
      "env": { "COMFYUI_DIR": "C:/ComfyUI_windows_portable/ComfyUI" }
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add comfyui --env COMFYUI_DIR=C:/ComfyUI_windows_portable/ComfyUI -- opendaemon-comfyui-mcp
```

**Cursor**: in `.cursor/mcp.json`, use the same block as Claude Desktop.

**OpenDaemon**: in `opendeamon.json`:

```json
{ "mcp": { "servers": { "comfyui": {
  "command": "opendaemon-comfyui-mcp",
  "env": { "COMFYUI_DIR": "C:/ComfyUI_windows_portable/ComfyUI", "OPENDAEMON_COMFY_TOOLS": "core" } } } } }
```

Then ask, for example: *"Make 3 short clips of the cat in C:/pics/cat.jpg and join them into one video."*

## Settings (environment variables)

| Variable | What it does | Default |
|---|---|---|
| `COMFYUI_DIR` | ComfyUI's folder (the one with `main.py`) | looked for in the usual places |
| `COMFYUI_URL` | ComfyUI's address | `http://127.0.0.1:8188` |
| `COMFYUI_PYTHON` | the Python that runs ComfyUI, used to start it and to install node packs | the portable build's own |
| `COMFY_MCP_DATA` | where videos, your workflows and settings are kept | `~/OpenDaemon-ComfyUI-MCP` |
| `OPENDAEMON_COMFY_TOOLS` | `full` (67 tools), `core` (15 tools) or a comma list of tool names | `full` |
| `GEMINI_API_KEY` / `NVIDIA_API_KEY` | a free vision model key, used to write prompts and to judge clips | none: those two tools say so |
| `FFMPEG` | a specific ffmpeg | found by itself (one comes with this package) |
| `PIPELINE_PORT` / `PIPELINE_URL` | the pipeline server's port | `8190` |
| `PIPELINE_AUTOSTART` | `0` = never start the pipeline server itself (for an app that runs it); `PIPELINE_DOWN_HINT` is then the message the model gets | `1` |

**Why `core`?** An AI app sends every tool's description to the model with every message. All 67 tools cost
about 7,000 tokens per message; the core 15 cost about 2,200. Small local models with a short context should use
`core`.

## Your own ComfyUI workflows

1. In ComfyUI, open your workflow, then **Workflow menu > Export (API)**. The normal save format can't be run
   over the API.
2. Put the file in `COMFY_MCP_DATA/workflows/`, or ask the assistant to keep it (`save_workflow`).
3. Ask for it by name. The assistant sees its inputs (`list_workflows`), checks it (`check_workflow`) and runs it
   (`run_workflow`).

Inputs are found automatically: the prompt and negative prompt, picture, video, audio, seed, steps, cfg, denoise,
width, height, length and fps. To choose the inputs yourself, rename a node in ComfyUI to `$name` (its main input)
or `$name.field`. Nodes with the same `$name` change together. A new random seed is used each run unless you
give one.

There's an example in `examples/workflows/ltx-image-to-video.json`; it's copied into your workflows folder on
first use.

## The tools

| Group | Tools |
|---|---|
| System | comfyui_status, start_comfyui, restart_comfyui, free_gpu_memory, get_logs, get_defaults, set_defaults |
| Models | list_models, download_model, download_status, cancel_download, search_models, check_model, add_model_file, remove_model |
| Pictures and prompts | upload_image, write_prompt, write_avoid_list, render_options |
| Render and queue | render, rerender, job_status, cancel, diagnose_job, estimate, render_batch, render_variations, queue, suggest_settings |
| Review and library | review_video, auto_fix, list_videos, video_info, delete_video |
| Editing | build_video, join_clips, trim_clip, extend_clip, change_speed, reverse_clip, loop_clip, boomerang, smooth_motion, upscale_clip, crop_zoom, color_adjust, fade, add_music, make_voice, add_voice, lipsync, add_captions, add_text, add_logo, title_card, end_card, reformat, export, thumbnail, make_gif, extract_audio, remove_audio |
| Your workflows | list_workflows, run_workflow, check_workflow, install_custom_node, save_workflow |

The server also gives the model a short guide to prompting video models (the resource `guide://prompting`).

## Safety

- Anything that deletes, installs or downloads a lot asks you first: the tool returns `needs_confirmation`, and the
  assistant must ask you before calling it again with `user_confirmed=true`.
- Node packs are programs written by other people. `install_custom_node` only takes GitHub links and always asks first.
- The model list says how much graphics memory each model needs and which one is best for your card. Files found
  with `search_models` can only be downloaded if they fit your card.
- Freeing GPU memory or restarting ComfyUI is refused while something renders.

## Licence

MIT, see [LICENSE](LICENSE). ComfyUI (GPL-3.0), the node packs and the models have their own licences.

Made by [OpenDaemon](https://open-daemon.com).
