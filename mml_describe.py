#!/usr/bin/env python
"""
Interactive Molmo → point → mark → refeed loop (+ local-area describe: crop / boxed / composite).

What it does
============
- Loads a Molmo MLLM and an initial image.
- Two interaction modes:
  • **Builder mode (default):** you type a *description*, then a *target*. The script auto-builds:
      "<description> Point to the <target>. Return ONLY a JSON object with fields \"x\" and \"y\" (floats with 3 decimals)."
  • **Free-prompt mode:** you type the full prompt yourself (original behavior).
- The model replies; we parse {"x": float, "y": float} as *percent-of-width/height*.
- A red dot is stamped at that percentage coordinate and saved to disk.
- That *new* (marked) image becomes the next-round input.

Sanity check (new)
==================
After each detected point (when a *target* is known), we crop around the point and ask:
  "Is <target> in this image? Answer \"YES\" or \"NO\" and briefly justify."
We print the raw response and log it. Box size is configurable via `/sanity_box <pct>`.

New helpers for local description (Molmo=single image)
=====================================================
Because Molmo accepts one image per call, we add a stitched **composite** that shows:
(A) an unmarked tight crop of the target region (precision), and
(B) the full original with a red rectangle (context), side-by-side with labels.
You can also run just the crop or just the boxed full image.

Commands
========
- `/mode builder` | `/mode prompt` — switch input mode.
- `/sanity on|off` — enable/disable the post-point sanity check.
- `/sanity_box <pct>` — set crop size used by sanity check (default 12.0).
- `/mark <x_pct> <y_pct>` — manually add a dot (percent coords) without querying the model.
- `/describe_crop <step|last> [box_pct]` — describe using only the tight crop centered at a stored point.
- `/describe_boxed <step|last> [box_pct]` — describe using only the original image with a red rectangle.
- `/describe_both <step|last> [box_pct]` — describe using a stitched composite (crop+boxed) with an instruction to focus on (A).
- `/describe_xy[_crop|_boxed|_both] <x_pct> <y_pct> [box_pct]`
- `/undo`, `/reset`, `/radius <px>`, `/image <path>`, `exit` or `quit`

Notes
=====
- Coordinates are clamped to [0, 100] percent.
- Dot radius defaults to 5 px.
- Output filenames carry step index and coordinates: `<stem>_step{N}_{x:.3f}_{y:.3f}<suffix>`
- **GPU memory caps default** to `8GiB,10GiB,10GiB,11GiB` (override with `--gpu-mems`).
- Crops use a small context pad (1.2× of the requested box) and are clamped to image bounds.

"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from accelerate import infer_auto_device_map


# ----------------------- Utility: prompt building -----------------------

FIXED_RETURN = 'Return ONLY a JSON object with fields "x" and "y" (floats with 3 decimals).'

def _endswith_punct(s: str) -> bool:
    return bool(s) and s.strip().endswith(('.', '!', '?'))

def build_point_prompt(description: str, target: str) -> str:
    description = description.strip()
    target = target.strip()
    if not _endswith_punct(description):
        description += '.'
    return f'{description} Point to the {target}. {FIXED_RETURN}'

# ----------------------- Utility: image marking -----------------------

def mark_image(
    path: Path,
    x_pct: float,
    y_pct: float,
    radius: int = 5,
    label: Optional[str] = None,
    outdir: Optional[Path] = None,
    step: Optional[int] = None,
) -> Path:
    """Draw a red dot at (x_pct, y_pct) on *path* and save a copy. Returns the new file path."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    x_pct = max(0.0, min(100.0, float(x_pct)))
    y_pct = max(0.0, min(100.0, float(y_pct)))
    x = int(round(w * x_pct / 100.0))
    y = int(round(h * y_pct / 100.0))

    draw = ImageDraw.Draw(img)
    draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=(255, 0, 0))

    if label:
        try:
            font = ImageFont.load_default()
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
            pad = 2
            box = [x + radius + 4, y - text_h - pad, x + radius + 4 + text_w + 2 * pad, y + pad]
            draw.rectangle(box, fill=(255, 255, 255))
            draw.text((box[0] + pad, box[1] + pad), label, fill=(0, 0, 0), font=font)
        except Exception:
            draw.text((x + radius + 4, y - 10), label, fill=(255, 255, 255))

    outdir = outdir or path.parent
    stem = path.stem
    suffix = path.suffix
    step_tag = f"_step{step}" if step is not None else ""
    out_name = f"{stem}{step_tag}_{x_pct:.3f}_{y_pct:.3f}{suffix}"
    out_path = outdir / out_name
    img.save(out_path)
    return out_path


# --------------------- Utility: parse model JSON ----------------------

_JSON_SNIP_RE = re.compile(r"\{.*?\}", re.DOTALL)

def parse_xy_from_text(text: str) -> Tuple[float, float]:
    """Extract x,y floats from model text with robust fallbacks."""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "x" in obj and "y" in obj:
            return float(obj["x"]), float(obj["y"])
    except Exception:
        pass

    for m in _JSON_SNIP_RE.finditer(text):
        snippet = m.group(0)
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict) and "x" in obj and "y" in obj:
                return float(obj["x"]), float(obj["y"])
        except Exception:
            continue

    x_match = re.search(r"\"x\"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    y_match = re.search(r"\"y\"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", text)
    if x_match and y_match:
        return float(x_match.group(1)), float(y_match.group(1))

    raise ValueError(f"Could not parse x,y from model output: {text!r}")


# --------------------- Local-area render helpers ---------------------

def _compute_box(w: int, h: int, x_pct: float, y_pct: float, box_pct: float, pad_scale: float = 1.2) -> Tuple[int,int,int,int]:
    """Return (left, top, right, bottom) for a square box centered on (x%,y%).
    side = box_pct% of min(w,h), then expanded by pad_scale (clamped to bounds)."""
    x_pct = max(0.0, min(100.0, float(x_pct)))
    y_pct = max(0.0, min(100.0, float(y_pct)))
    cx = w * x_pct / 100.0
    cy = h * y_pct / 100.0
    base_side = max(2.0, (min(w, h) * max(0.1, float(box_pct)) / 100.0))
    side = base_side * float(pad_scale)
    half = side / 2.0
    left = int(round(max(0, cx - half)))
    top = int(round(max(0, cy - half)))
    right = int(round(min(w, cx + half)))
    bottom = int(round(min(h, cy + half)))
    # Ensure non-empty
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def draw_box_on_original(original_path: Path, x_pct: float, y_pct: float, box_pct: float = 10.0, border_px: int = 3, outdir: Optional[Path] = None, step: Optional[int] = None) -> Path:
    """Draw a red rectangle on the **original** image and save copy."""
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    l, t, r, b = _compute_box(w, h, x_pct, y_pct, box_pct, pad_scale=1.0)
    draw = ImageDraw.Draw(img)
    draw.rectangle([l, t, r, b], outline=(255, 0, 0), width=int(border_px))
    outdir = outdir or original_path.parent
    stem = original_path.stem
    suffix = original_path.suffix
    step_tag = f"_step{step}" if step is not None else ""
    out_name = f"{stem}{step_tag}_box_{x_pct:.3f}_{y_pct:.3f}_{box_pct:.2f}{suffix}"
    out_path = outdir / out_name
    img.save(out_path)
    return out_path


def crop_around_point(original_path: Path, x_pct: float, y_pct: float, box_pct: float = 10.0, outdir: Optional[Path] = None, step: Optional[int] = None, pad_scale: float = 1.2, min_side_px: int = 64, max_side_px: int = 1024) -> Path:
    """Save an **unmarked** crop centered on (x%,y%) with small context pad."""
    img = Image.open(original_path).convert("RGB")
    w, h = img.size
    l, t, r, b = _compute_box(w, h, x_pct, y_pct, box_pct, pad_scale=pad_scale)
    # Enforce min/max side
    side = max(r - l, b - t)
    if side < min_side_px:
        grow = (min_side_px - side) / 2.0
        l = int(max(0, l - grow)); t = int(max(0, t - grow))
        r = int(min(w, r + grow)); b = int(min(h, b + grow))
    if side > max_side_px:
        shrink = (side - max_side_px) / 2.0
        l = int(l + shrink); t = int(t + shrink)
        r = int(r - shrink); b = int(b - shrink)
    crop = img.crop((l, t, r, b))
    outdir = outdir or original_path.parent
    stem = original_path.stem
    suffix = original_path.suffix
    step_tag = f"_step{step}" if step is not None else ""
    out_name = f"{stem}{step_tag}_crop_{x_pct:.3f}_{y_pct:.3f}_{box_pct:.2f}{suffix}"
    out_path = outdir / out_name
    crop.save(out_path)
    return out_path


def make_composite(crop_path: Path, boxed_path: Path, label_a: str = "A", label_b: str = "B") -> Path:
    """Stitch crop (A) and boxed-full (B) into one side-by-side image with labels."""
    crop = Image.open(crop_path).convert("RGB")
    boxed = Image.open(boxed_path).convert("RGB")

    # Normalize heights
    h_target = 512  # reasonable working height
    def scale_to_height(im: Image.Image, h: int) -> Image.Image:
        w, hh = im.size
        if hh == h:
            return im
        new_w = int(round(w * h / hh))
        return im.resize((new_w, h), Image.LANCZOS)

    crop = scale_to_height(crop, h_target)
    boxed = scale_to_height(boxed, h_target)

    gap = 12
    divider = 2
    out_w = crop.width + gap + divider + gap + boxed.width
    out_h = h_target

    canvas = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    x0 = 0
    canvas.paste(crop, (x0, 0))
    x1 = x0 + crop.width + gap
    # divider
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([x1, 0, x1 + divider, out_h], fill=(200, 200, 200))
    x2 = x1 + divider + gap
    canvas.paste(boxed, (x2, 0))

    # Labels
    try:
        font = ImageFont.load_default()
        draw.text((4, 4), label_a, fill=(0, 0, 0), font=font)
        draw.text((x2 + 4, 4), label_b, fill=(0, 0, 0), font=font)
    except Exception:
        pass

    out_path = crop_path.with_name(crop_path.stem + "__COMPOSITE__" + boxed_path.suffix)
    canvas.save(out_path)
    return out_path


# --------------------------- Molmo wrapper ---------------------------

@dataclass
class MolmoConfig:
    model_id: str
    max_new_tokens: int = 64
    stop_strings: Tuple[str, ...] = ("<|endoftext|>",)


class MolmoSession:
    def __init__(self, cfg: MolmoConfig, gpu_mems: Optional[Tuple[str, ...]] = None):
        self.cfg = cfg
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="meta",
        )

        if gpu_mems is None:
            device_map = infer_auto_device_map(base_model)
        else:
            max_memory: Dict[int, str] = {i: mem for i, mem in enumerate(gpu_mems)}
            device_map = infer_auto_device_map(base_model, max_memory=max_memory, no_split_module_classes=[])

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
        )

    @torch.inference_mode()
    def predict_xy(self, image: Image.Image, prompt: str) -> Tuple[str, float, float]:
        inputs = self.processor.process(images=image, text=prompt, return_tensors="pt")
        inputs = {k: (v.unsqueeze(0) if v.dim() == 0 or v.shape[0] != 1 else v) for k, v in inputs.items()}
        try:
            device = getattr(self.model, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        gen_cfg = GenerationConfig(max_new_tokens=self.cfg.max_new_tokens, stop_strings=list(self.cfg.stop_strings))
        output = self.model.generate_from_batch(inputs, gen_cfg, tokenizer=self.processor.tokenizer)
        input_len = inputs["input_ids"].size(1)
        generated_tokens = output[0, input_len:]
        raw = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        x, y = parse_xy_from_text(raw)
        return raw, x, y

    @torch.inference_mode()
    def describe(self, image: Image.Image, prompt: str) -> str:
        inputs = self.processor.process(images=image, text=prompt, return_tensors="pt")
        inputs = {k: (v.unsqueeze(0) if v.dim() == 0 or v.shape[0] != 1 else v) for k, v in inputs.items()}
        try:
            device = getattr(self.model, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
        gen_cfg = GenerationConfig(max_new_tokens=self.cfg.max_new_tokens, stop_strings=list(self.cfg.stop_strings))
        output = self.model.generate_from_batch(inputs, gen_cfg, tokenizer=self.processor.tokenizer)
        input_len = inputs["input_ids"].size(1)
        generated_tokens = output[0, input_len:]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ------------------------------ Main REPL ----------------------------

def run_loop(
    image_path: Path,
    model_id: str,
    outdir: Optional[Path] = None,
    dot_radius: int = 5,
    max_new_tokens: int = 64,
    gpu_mems: Optional[Tuple[str, ...]] = None,
    label_steps: bool = True,
    builder_mode_default: bool = True,
    sanity_check: bool = True,
    sanity_box_pct: float = 12.0,
):
    outdir = outdir or image_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    log_path = outdir / f"{image_path.stem}_molmo_loop.log.jsonl"

    cfg = MolmoConfig(model_id=model_id, max_new_tokens=max_new_tokens)
    session = MolmoSession(cfg, gpu_mems=gpu_mems)

    print("Model loaded.")
    print("Commands: /mode builder | /mode prompt | /sanity on|off | /sanity_box PCT | /mark x y | /undo | /reset | /radius N | /image PATH | /describe_* ... | exit")
    if builder_mode_default:
        print("Input mode: BUILDER. Type a description, press Enter, then provide a target.")
    else:
        print("Input mode: PROMPT. Type the full prompt.")

    original = image_path.resolve()
    current = original
    history: List[Path] = []
    points: Dict[int, Tuple[float, float]] = {}
    targets: Dict[int, str] = {}
    step = 0
    builder_mode = builder_mode_default

    while True:
        try:
            user_input = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        lower = user_input.lower()
        if lower in {"exit", "quit"}:
            break

        # ---- Mode + sanity controls ----
        if lower.startswith("/mode "):
            mode = lower.split(maxsplit=1)[1]
            if mode == "builder":
                builder_mode = True
                print("Switched to BUILDER mode (description → target → auto prompt).")
            elif mode == "prompt":
                builder_mode = False
                print("Switched to PROMPT mode (type full prompt).")
            else:
                print("Usage: /mode builder | /mode prompt")
            continue

        if lower.startswith("/sanity "):
            val = lower.split(maxsplit=1)[1]
            if val == "on":
                sanity_check = True; print("Sanity check: ON")
            elif val == "off":
                sanity_check = False; print("Sanity check: OFF")
            else:
                print("Usage: /sanity on|off")
            continue

        if lower.startswith("/sanity_box "):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    sanity_box_pct = float(parts[1]); print(f"Sanity crop box set to {sanity_box_pct:.2f}% of min(w,h)")
                except Exception as e:
                    print(f"/sanity_box error: {e}")
            else:
                print("Usage: /sanity_box <pct>")
            continue

        # ---- Existing commands ----
        if lower.startswith("/mark "):
            parts = user_input.split()
            if len(parts) >= 3:
                try:
                    x_pct = float(parts[1]); y_pct = float(parts[2])
                    step += 1
                    label = str(step) if label_steps else None
                    out = mark_image(current, x_pct, y_pct, radius=dot_radius, label=label, outdir=outdir, step=step)
                    history.append(out)
                    points[step] = (round(x_pct, 3), round(y_pct, 3))
                    current = out
                    _append_log(log_path, {"ts": _now(), "step": step, "mode": "manual", "prompt": None, "x": round(x_pct,3), "y": round(y_pct,3), "output": str(out)})
                    print(f"Saved marked image → {out}")
                    continue
                except Exception as e:
                    print(f"/mark error: {e}")
                    continue
            else:
                print("Usage: /mark <x_pct> <y_pct>")
                continue

        if lower.startswith("/radius "):
            parts = user_input.split()
            if len(parts) >= 2:
                try:
                    dot_radius = int(float(parts[1])); print(f"Dot radius set to {dot_radius}px")
                except Exception as e:
                    print(f"/radius error: {e}")
            else:
                print("Usage: /radius <pixels>")
            continue

        if lower.startswith("/image "):
            p = Path(user_input.split(maxsplit=1)[1].strip())
            if not p.exists():
                print(f"Path not found: {p}")
                continue
            original = p.resolve(); current = original
            history.clear(); points.clear(); targets.clear(); step = 0
            print(f"Switched base image → {current}")
            continue

        if lower == "/reset":
            current = original
            history.clear(); points.clear(); targets.clear(); step = 0
            print("Reset to original image.")
            continue

        if lower == "/undo":
            if history:
                history.pop()
                if step in points: points.pop(step, None)
                if step in targets: targets.pop(step, None)
                step -= 1
                current = history[-1] if history else original
                print(f"Undid last step. Current image → {current}")
            else:
                print("Nothing to undo.")
            continue

        # ---- Describe variants (unchanged) ----
        def _resolve_step_or_last(token: str) -> Optional[int]:
            if token == "last":
                return max(points.keys()) if points else None
            try:
                return int(token)
            except Exception:
                return None

        def _describe_core(x_pct: float, y_pct: float, box_pct: float, mode: str):
            try:
                # Build resources
                boxed = draw_box_on_original(original, x_pct, y_pct, box_pct=box_pct, outdir=outdir)
                crop = crop_around_point(original, x_pct, y_pct, box_pct=box_pct, outdir=outdir)
                if mode == "both":
                    comp = make_composite(crop, boxed)
                    img = Image.open(comp).convert("RGB")
                    prompt = (
                        "You are given a stitched image with two panels. "
                        "Panel A (left) is a tight crop of the target region. Panel B (right) is the full image with a red box showing the same region. "
                        "Describe ONLY what is inside Panel A. Use Panel B only as context if needed."
                    )
                    text = session.describe(img, prompt)
                    _append_log(log_path, {"ts": _now(), "mode": "describe_both", "x": round(x_pct,3), "y": round(y_pct,3), "box_pct": box_pct, "boxed_image": str(boxed), "crop_image": str(crop), "composite": str(comp), "text": text})
                    print(f"Describe(both, box={box_pct}%):\n{text}\nSaved composite → {comp}")
                elif mode == "crop":
                    img = Image.open(crop).convert("RGB")
                    prompt = "Describe the area in this image."
                    text = session.describe(img, prompt)
                    _append_log(log_path, {"ts": _now(), "mode": "describe_crop", "x": round(x_pct,3), "y": round(y_pct,3), "box_pct": box_pct, "crop_image": str(crop), "text": text})
                    print(f"Describe(crop, box={box_pct}%):\n{text}\nSaved crop → {crop}")
                else:  # boxed
                    img = Image.open(boxed).convert("RGB")
                    prompt = "Describe the area inside the red box."
                    text = session.describe(img, prompt)
                    _append_log(log_path, {"ts": _now(), "mode": "describe_boxed", "x": round(x_pct,3), "y": round(y_pct,3), "box_pct": box_pct, "boxed_image": str(boxed), "text": text})
                    print(f"Describe(boxed, box={box_pct}%):\n{text}\nSaved boxed image → {boxed}")
            except Exception as e:
                print(f"Describe error: {e}")

        # step-based
        if lower.startswith("/describe_crop ") or lower.startswith("/describe_boxed ") or lower.startswith("/describe_both "):
            parts = user_input.split()
            mode = "crop" if lower.startswith("/describe_crop ") else ("boxed" if lower.startswith("/describe_boxed ") else "both")
            if len(parts) >= 2:
                target_step = parts[1]; box_pct = float(parts[2]) if len(parts) >= 3 else 12.0
                s = _resolve_step_or_last(target_step)
                if s is None or s not in points:
                    print("No stored point for that step.")
                else:
                    x_pct, y_pct = points[s]
                    _describe_core(x_pct, y_pct, box_pct, mode)
            else:
                print("Usage: /describe_crop|_boxed|_both <step|last> [box_pct]")
            continue

        # xy-based
        if lower.startswith("/describe_xy"):
            parts = user_input.split()
            mode = "both"
            if lower.startswith("/describe_xy_crop"):
                mode = "crop"
            elif lower.startswith("/describe_xy_boxed"):
                mode = "boxed"
            if len(parts) >= 3:
                try:
                    x_pct = float(parts[1]); y_pct = float(parts[2])
                    box_pct = float(parts[3]) if len(parts) >= 4 else 12.0
                    _describe_core(x_pct, y_pct, box_pct, mode)
                except Exception as e:
                    print(f"/describe_xy error: {e}")
            else:
                print("Usage: /describe_xy[_crop|_boxed|_both] <x_pct> <y_pct> [box_pct]")
            continue

        # ---------------- Normal round: builder or free prompt ----------------
        try:
            img = Image.open(current).convert("RGB")
        except Exception as e:
            print(f"Failed to open current image {current}: {e}")
            continue

        try:
            if builder_mode:
                # Interpret this line as the *description*; then ask for target.
                description = user_input
                if not description:
                    print("Please enter a description (or use /mode prompt to type a full prompt).")
                    continue
                target = input("target> ").strip()
                if not target:
                    print("Target cannot be empty.")
                    continue
                final_prompt = build_point_prompt(description, target)
                raw, x_pct, y_pct = session.predict_xy(img, final_prompt)
                used_target = target
                used_prompt = final_prompt
            else:
                # Free prompt: user_input is the full prompt; target unknown
                raw, x_pct, y_pct = session.predict_xy(img, user_input)
                used_target = None
                used_prompt = user_input
        except Exception as e:
            print(f"Model/pipeline error: {e}")
            continue

        x_pct = max(0.0, min(100.0, float(x_pct)))
        y_pct = max(0.0, min(100.0, float(y_pct)))
        xr = round(x_pct, 3); yr = round(y_pct, 3)

        step += 1
        label = str(step) if label_steps else None
        out = mark_image(current, xr, yr, radius=dot_radius, label=label, outdir=outdir, step=step)
        history.append(out)
        points[step] = (xr, yr)
        current = out
        if used_target is not None:
            targets[step] = used_target

        log_entry = {"ts": _now(), "step": step, "mode": "model", "prompt": used_prompt, "raw_text": raw, "x": xr, "y": yr, "output": str(out)}
        if used_target is not None:
            log_entry["target"] = used_target

        print(f"Model → x={xr:.3f}%, y={yr:.3f}% | saved → {out}")

        # ---------------- Sanity check (crop + ask if target is present) ----------------
        if sanity_check and used_target:
            try:
                crop_path = crop_around_point(original, xr, yr, box_pct=sanity_box_pct, outdir=outdir, step=step)
                crop_img = Image.open(crop_path).convert("RGB")
                sanity_prompt = f'Is {used_target} in this image? Answer "YES" or "NO". If "YES", additionally provide a nearby landmark relative to the object in the image and describe its relation to {used_target}'
                sanity_text = session.describe(crop_img, sanity_prompt)
                log_entry.update({"sanity": {"box_pct": sanity_box_pct, "crop_image": str(crop_path), "prompt": sanity_prompt, "text": sanity_text}})
                print(f'Sanity({used_target}, box={sanity_box_pct}%): {sanity_text}\nSaved sanity crop → {crop_path}')
            except Exception as e:
                print(f"Sanity check error: {e}")

        _append_log(log_path, log_entry)

    print("Done.")


# ------------------------------ Helpers ------------------------------

def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _append_log(path: Path, entry: Dict):
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Log write failed: {e}")


# ------------------------------ CLI entry ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Molmo point→mark→refeed interactive loop")
    p.add_argument("image", type=Path, help="Path to the starting image")
    p.add_argument("--model-id", default="/workspace/Molmo-7B-D-0924", help="HF model id or local path")
    p.add_argument("--outdir", type=Path, default=None, help="Directory to save marked images and log")
    p.add_argument("--dot-radius", type=int, default=5, help="Dot radius in pixels")
    p.add_argument("--max-new-tokens", type=int, default=64, help="Max new tokens for generation")
    p.add_argument(
        "--gpu-mems",
        type=str,
        default="8GiB,10GiB,10GiB,11GiB",
        help="Comma-separated per-GPU memory caps like 8GiB,10GiB,10GiB,11GiB (required on your setup)",
    )
    p.add_argument("--no-step-labels", action="store_true", help="Disable small step-number labels by dots")
    p.add_argument("--free-prompt", action="store_true", help="Start in free-prompt mode instead of builder mode")
    p.add_argument("--no-sanity", action="store_true", help="Disable the sanity check after each point")
    p.add_argument("--sanity-box-pct", type=float, default=12.0, help="Crop size (percent of min dim) for sanity check")
    return p.parse_args()


def main():
    args = parse_args()
    gpu_mems: Optional[Tuple[str, ...]] = None
    if args.gpu_mems:
        gpu_mems = tuple(x.strip() for x in args.gpu_mems.split(",") if x.strip())

    run_loop(
        image_path=args.image,
        model_id=args.model_id,
        outdir=args.outdir,
        dot_radius=args.dot_radius,
        max_new_tokens=args.max_new_tokens,
        gpu_mems=gpu_mems,
        label_steps=not args.no_step_labels,
        builder_mode_default=not args.free_prompt,
        sanity_check=not args.no_sanity,
        sanity_box_pct=float(args.sanity_box_pct),
    )


if __name__ == "__main__":
    main()
