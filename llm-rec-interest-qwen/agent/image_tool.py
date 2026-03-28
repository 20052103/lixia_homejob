# -*- coding: utf-8 -*-
"""
Image reading skill.

Flow:
  1. A tkinter dialog pops up offering "Paste from Clipboard" or "Browse File".
  2. The image is captured and base64-encoded.
  3. Try to describe the image via LM Studio vision API (OpenAI-compatible).
  4. If the model doesn't support vision, fall back to pytesseract OCR.
"""
from __future__ import annotations

import base64
import io
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional

from PIL import Image, ImageGrab


# ============================================================
# GUI dialog — returns a PIL Image or None
# ============================================================

def _pick_image_gui() -> Optional[Image.Image]:
    """
    Open a small Tkinter window asking the user to paste from clipboard
    or browse for a file. Returns a PIL Image, or None if cancelled.
    """
    result: dict = {"image": None}

    root = tk.Tk()
    root.title("📷 Provide Image")
    root.resizable(False, False)
    root.attributes("-topmost", True)

    # Centre on screen
    root.update_idletasks()
    w, h = 340, 150
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

    tk.Label(root, text="How would you like to provide the image?",
             font=("Segoe UI", 10), pady=12).pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack()

    def _from_clipboard():
        try:
            img = ImageGrab.grabclipboard()
            if img is None:
                messagebox.showerror("No image", "No image found in clipboard.\nCopy an image first.")
                return
            if not isinstance(img, Image.Image):
                # grabclipboard can return a list of file paths
                if isinstance(img, list) and img:
                    img = Image.open(img[0])
                else:
                    messagebox.showerror("No image", "Clipboard content is not an image.")
                    return
            result["image"] = img.convert("RGB")
            root.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _from_file():
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.tiff"), ("All files", "*.*")],
        )
        if path:
            result["image"] = Image.open(path).convert("RGB")
            root.destroy()

    def _cancel():
        root.destroy()

    tk.Button(btn_frame, text="📋 Paste from Clipboard", width=22,
              command=_from_clipboard, bg="#4A90D9", fg="white",
              font=("Segoe UI", 10)).grid(row=0, column=0, padx=6, pady=4)
    tk.Button(btn_frame, text="📁 Browse File", width=16,
              command=_from_file, bg="#5CB85C", fg="white",
              font=("Segoe UI", 10)).grid(row=0, column=1, padx=6, pady=4)
    tk.Button(root, text="Cancel", command=_cancel,
              font=("Segoe UI", 9)).pack(pady=6)

    root.mainloop()
    return result["image"]


# ============================================================
# Image → base64
# ============================================================

def _image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ============================================================
# Vision via LM Studio (OpenAI-compatible)
# ============================================================

def _describe_via_vision(img: Image.Image, prompt: str, base_url: str, api_key: str, model: str) -> Optional[str]:
    """Send image to LM Studio vision endpoint. Returns description or None if unsupported."""
    try:
        from openai import OpenAI
        b64 = _image_to_base64(img)
        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        err = str(e).lower()
        # Vision not supported by this model
        if any(k in err for k in ["vision", "image", "multimodal", "not support", "invalid"]):
            print(f"[image_tool] Vision not supported by model: {e}")
            return None
        print(f"[image_tool] Vision API error: {e}")
        return None


# ============================================================
# OCR fallback via pytesseract
# ============================================================

def _ocr_image(img: Image.Image) -> str:
    try:
        import pytesseract
        text = pytesseract.image_to_string(img).strip()
        return text if text else "(No text found by OCR)"
    except Exception as e:
        return f"(OCR failed: {e}. Is Tesseract installed? https://github.com/UB-Mannheim/tesseract/wiki)"


# ============================================================
# Public API
# ============================================================

def read_image_interactive(
    prompt: str = "Describe this image in detail. If it contains text, transcribe it.",
    base_url: str = "http://127.0.0.1:1234/v1",
    api_key: str = "lm-studio",
    model: str = "",
) -> str:
    """
    Show GUI to get an image, then describe it via vision LLM or OCR.
    Returns the description string.
    """
    print("[image_tool] Opening image picker dialog...")
    img = _pick_image_gui()

    if img is None:
        return "Image reading cancelled."

    print(f"[image_tool] Image captured: {img.size[0]}x{img.size[1]} px")

    # Try vision model first
    if base_url and model:
        print(f"[image_tool] Trying vision via {model}...")
        description = _describe_via_vision(img, prompt, base_url, api_key, model)
        if description:
            print("[image_tool] Vision succeeded.")
            return description
        print("[image_tool] Vision failed — falling back to OCR.")

    # OCR fallback
    print("[image_tool] Running OCR...")
    ocr_text = _ocr_image(img)
    return f"[OCR result]\n{ocr_text}"
