"""
Dental Surface Reconstruction App  v4.0
=========================================
KEY INSIGHT from v1-v3 failures:
  Auto-detecting the gap from diff images fails when reference and target
  are taken from different angles/crops. The ONLY reliable solution is to
  let the user paint the gap region directly on the target image.

Pipeline:
  1. User paints over the gap on the target image (ImageEditor with brush)
  2. System extracts that painted region as the inpaint mask
  3. Fills using: symmetry patch from reference → colour-matched → Poisson blend
  4. Refines with OpenCV Telea inpainting on the boundary seam
  5. Optional SD inpainting on GPU
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import warnings

warnings.filterwarnings("ignore")

# ── Optional diffusion ────────────────────────────────────────────────────────
DIFFUSION_AVAILABLE = False
pipe = None
try:
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    DIFFUSION_AVAILABLE = True
    print(f"[INFO] Diffusion ready on {device}.")
except Exception as e:
    print(f"[INFO] Classical pipeline ({type(e).__name__}).")


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

def to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def fit(img: Image.Image, size: int = 768) -> Image.Image:
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    return img.resize((nw, nh), Image.LANCZOS)

def match_hw(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    return cv2.resize(src, (w, h), interpolation=cv2.INTER_LANCZOS4)


# ══════════════════════════════════════════════════════════════════════════════
#  EXTRACT MASK FROM IMAGEEDITOR OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def extract_painted_mask(editor_value: dict) -> np.ndarray | None:
    """
    Gradio ImageEditor returns a dict with keys:
      'background' : original image (numpy RGBA or RGB)
      'layers'     : list of painted layers (numpy RGBA)
      'composite'  : merged result (numpy RGBA or RGB)

    We detect the user-painted strokes by finding non-transparent pixels
    in the layers that differ from the background.
    Returns a uint8 binary mask (255 = painted gap).
    """
    if editor_value is None:
        return None

    layers = editor_value.get("layers", [])
    bg     = editor_value.get("background")

    if not layers and bg is None:
        return None

    # Merge all paint layers
    h, w = None, None
    if bg is not None:
        bg_arr = np.array(bg)
        h, w = bg_arr.shape[:2]
    
    if h is None and layers:
        h, w = np.array(layers[0]).shape[:2]

    if h is None:
        return None

    combined_alpha = np.zeros((h, w), dtype=np.float32)

    for layer in layers:
        if layer is None:
            continue
        la = np.array(layer)
        if la.ndim == 3 and la.shape[2] == 4:
            # Alpha channel of paint layer = where user painted
            alpha = la[:, :, 3].astype(np.float32) / 255.0
            combined_alpha = np.maximum(combined_alpha, alpha)
        elif la.ndim == 3 and la.shape[2] == 3:
            # RGB layer: find painted pixels as bright non-background areas
            if bg is not None:
                bg_rgb  = np.array(Image.fromarray(bg_arr).convert("RGB"))
                la_rgb  = la[:, :, :3]
                diff    = np.abs(la_rgb.astype(float) - bg_rgb.astype(float)).mean(axis=2)
                alpha   = (diff > 20).astype(np.float32)
                combined_alpha = np.maximum(combined_alpha, alpha)

    mask = (combined_alpha > 0.1).astype(np.uint8) * 255

    # If nothing was painted, try composite vs background diff
    if mask.sum() == 0 and bg is not None:
        comp = editor_value.get("composite")
        if comp is not None:
            comp_arr = np.array(comp)
            if comp_arr.shape[2] >= 3:
                bg_rgb   = np.array(Image.fromarray(bg_arr).convert("RGB"))
                comp_rgb = comp_arr[:, :, :3]
                diff     = np.abs(comp_rgb.astype(float) - bg_rgb.astype(float)).mean(axis=2)
                mask     = (diff > 15).astype(np.uint8) * 255

    if mask.sum() == 0:
        return None

    # Clean up mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    mask = cv2.dilate(mask, k, iterations=2)
    return mask


# ══════════════════════════════════════════════════════════════════════════════
#  ALIGN REFERENCE TO TARGET
# ══════════════════════════════════════════════════════════════════════════════

def align(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    g1 = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(4000)
    kp1, d1 = orb.detectAndCompute(g1, None)
    kp2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(kp1) < 8 or len(kp2) < 8:
        return match_hw(src, dst)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(d1, d2), key=lambda m: m.distance)[:300]
    if len(matches) < 8:
        return match_hw(src, dst)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    if H is None:
        return match_hw(src, dst)
    h, w = dst.shape[:2]
    return cv2.warpPerspective(src, H, (w, h),
                                flags=cv2.INTER_LANCZOS4,
                                borderMode=cv2.BORDER_REPLICATE)


# ══════════════════════════════════════════════════════════════════════════════
#  COLOUR MATCH  (LAB statistics transfer)
# ══════════════════════════════════════════════════════════════════════════════

def colour_match_lab(patch: np.ndarray,
                     canvas: np.ndarray,
                     mask: np.ndarray) -> np.ndarray:
    """Match patch colours to the surrounding region of canvas at the mask."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    ring = cv2.dilate(mask, k, iterations=2)
    ring = cv2.bitwise_and(ring, cv2.bitwise_not(mask))

    surround = canvas[ring > 0]
    if len(surround) < 30:
        return patch

    # Work in LAB
    def to_lab_flat(bgr_arr):
        img = bgr_arr.reshape(1, -1, 3).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    s_lab = to_lab_flat(surround)
    p_lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    flat  = p_lab.reshape(-1, 3)

    s_mean, s_std = s_lab.mean(0), s_lab.std(0) + 1e-6
    p_mean, p_std = flat.mean(0), flat.std(0) + 1e-6

    matched = (flat - p_mean) * (s_std / p_std) + s_mean
    matched = np.clip(matched, 0, 255).reshape(p_lab.shape).astype(np.uint8)
    return cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  SYMMETRY PATCH  (mirror tooth from opposite side)
# ══════════════════════════════════════════════════════════════════════════════

def get_symmetry_patch(source: np.ndarray,
                       mask: np.ndarray,
                       notes: list):
    H, W = source.shape[:2]
    mid  = W // 2
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    gcx, gcy = float(xs.mean()), float(ys.mean())
    gw = int(xs.max() - xs.min()) + 24
    gh = int(ys.max() - ys.min()) + 24

    mirror_cx = 2 * mid - gcx

    mx1 = int(np.clip(mirror_cx - gw // 2, 0, W - 1))
    my1 = int(np.clip(gcy        - gh // 2, 0, H - 1))
    mx2 = int(np.clip(mirror_cx + gw // 2, 0, W))
    my2 = int(np.clip(gcy        + gh // 2, 0, H))

    if mx2 - mx1 < 10 or my2 - my1 < 10:
        notes.append("⚠️  Mirror region out of bounds.")
        return None

    patch = cv2.flip(source[my1:my2, mx1:mx2].copy(), 1)  # horizontal flip

    # Target paste region
    px1 = int(np.clip(gcx - gw // 2, 0, W - 1))
    py1 = int(np.clip(gcy - gh // 2, 0, H - 1))
    px2 = int(np.clip(gcx + gw // 2, 0, W))
    py2 = int(np.clip(gcy + gh // 2, 0, H))
    dw, dh = px2 - px1, py2 - py1

    if dw < 4 or dh < 4:
        return None

    patch = cv2.resize(patch, (dw, dh), interpolation=cv2.INTER_LANCZOS4)
    cmask = np.ones((dh, dw), dtype=np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cmask = cv2.erode(cmask, k, iterations=2)

    notes.append(f"🔄 Symmetry: gap cx={gcx:.0f} → mirror cx={mirror_cx:.0f}")
    return patch, cmask, px1, py1


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE COPY PATCH  (direct copy from aligned reference at mask location)
# ══════════════════════════════════════════════════════════════════════════════

def get_ref_patch(ref_cv: np.ndarray, mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = 10
    H, W = ref_cv.shape[:2]
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(W, int(xs.max()) + pad)
    y2 = min(H, int(ys.max()) + pad)
    patch = ref_cv[y1:y2, x1:x2].copy()
    cmask = np.ones((y2 - y1, x2 - x1), dtype=np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cmask = cv2.erode(cmask, k, iterations=2)
    return patch, cmask, x1, y1


# ══════════════════════════════════════════════════════════════════════════════
#  SEAMLESS PASTE
# ══════════════════════════════════════════════════════════════════════════════

def seamless_paste(canvas: np.ndarray,
                   patch: np.ndarray,
                   cmask: np.ndarray,
                   px: int, py: int,
                   notes: list) -> np.ndarray:
    H, W = canvas.shape[:2]
    ph, pw = patch.shape[:2]
    pw = min(pw, W - px)
    ph = min(ph, H - py)
    if pw <= 2 or ph <= 2:
        return canvas

    patch  = patch[:ph, :pw]
    cmask  = cmask[:ph, :pw]

    src    = np.zeros_like(canvas)
    src[py:py+ph, px:px+pw] = patch

    full_m = np.zeros((H, W), dtype=np.uint8)
    full_m[py:py+ph, px:px+pw] = cmask

    cx = int(np.clip(px + pw // 2, 5, W - 5))
    cy = int(np.clip(py + ph // 2, 5, H - 5))

    try:
        result = cv2.seamlessClone(src, canvas, full_m, (cx, cy), cv2.NORMAL_CLONE)
        notes.append("  ✅ Poisson seamless clone OK.")
        return result
    except Exception as e:
        notes.append(f"  ⚠️  Poisson failed ({e}) — feather blend.")
        feather = cv2.GaussianBlur(full_m.astype(np.float32) / 255.0, (41, 41), 0)
        f3 = feather[:, :, np.newaxis]
        return np.clip(f3 * src + (1 - f3) * canvas, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
#  SEAM REFINEMENT
# ══════════════════════════════════════════════════════════════════════════════

def refine_seam(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = cv2.erode(mask, k, iterations=3)
    seam  = cv2.bitwise_xor(mask, inner)
    if int((seam > 0).sum()) < 4:
        return img
    return cv2.inpaint(img, seam, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


# ══════════════════════════════════════════════════════════════════════════════
#  OPENCV-ONLY INPAINT FALLBACK  (when no patch is available)
# ══════════════════════════════════════════════════════════════════════════════

def opencv_inpaint_fill(canvas: np.ndarray, mask: np.ndarray,
                        notes: list) -> np.ndarray:
    notes.append("🔧 OpenCV Navier-Stokes inpaint fill …")
    result = cv2.inpaint(canvas, mask, inpaintRadius=16, flags=cv2.INPAINT_NS)
    notes.append("  ✅ OpenCV inpaint done.")
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RECONSTRUCT
# ══════════════════════════════════════════════════════════════════════════════

def reconstruct(ref1_np, ref2_np, editor_value):
    notes = []

    if ref1_np is None or ref2_np is None:
        return None, "⚠️  Please upload both reference images."
    if editor_value is None:
        return None, "⚠️  Please upload the target image and paint over the gap."

    # ── Get base target image ─────────────────────────────────────────────────
    bg = editor_value.get("background")
    if bg is None:
        return None, "⚠️  No target image found in editor. Please upload it first."

    try:
        target_pil = Image.fromarray(np.array(bg)).convert("RGB")
        ref1_pil   = Image.fromarray(ref1_np).convert("RGB")
        ref2_pil   = Image.fromarray(ref2_np).convert("RGB")
    except Exception as e:
        return None, f"❌ Image error: {e}"

    notes.append("✅ Images loaded.")
    orig_size = target_pil.size

    # ── Extract painted mask ──────────────────────────────────────────────────
    notes.append("🖌️  Extracting painted gap mask …")
    raw_mask = extract_painted_mask(editor_value)

    if raw_mask is None or int((raw_mask > 0).sum()) < 50:
        return None, ("⚠️  No gap region detected.\n\n"
                      "Please paint (draw) directly over the missing tooth gap "
                      "on the target image using the brush tool, then click Run.")

    gap_px = int((raw_mask > 0).sum())
    notes.append(f"🎯 Painted mask: {gap_px:,} px  "
                 f"({100 * gap_px / raw_mask.size:.1f}% of frame)")

    # ── Resize everything to working size ─────────────────────────────────────
    SIZE     = 768
    target_s = fit(target_pil, SIZE)
    ref1_s   = fit(ref1_pil,   SIZE)
    ref2_s   = fit(ref2_pil,   SIZE)

    # Scale mask to match working size
    tw, th = target_s.size
    mask = cv2.resize(raw_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.dilate(mask, k, iterations=1)

    target_cv = to_cv(target_s)
    ref1_cv   = to_cv(ref1_s)
    ref2_cv   = to_cv(ref2_s)

    # ── Align references to target ────────────────────────────────────────────
    notes.append("📐 Aligning references to target …")
    try:
        ref1_a = align(ref1_cv, target_cv)
        ref2_a = align(ref2_cv, target_cv)
        notes.append("✅ Alignment done.")
    except Exception as e:
        notes.append(f"⚠️  Alignment skipped ({e}).")
        ref1_a = match_hw(ref1_cv, target_cv)
        ref2_a = match_hw(ref2_cv, target_cv)

    # Best fill reference = brighter inside the gap (more tooth data)
    s1 = float(ref1_a[mask > 0].mean()) if mask.sum() > 0 else 0
    s2 = float(ref2_a[mask > 0].mean()) if mask.sum() > 0 else 0
    best_ref = ref1_a if s1 >= s2 else ref2_a
    notes.append(f"🧠 Fill ref: {'Ref 1' if s1>=s2 else 'Ref 2'} "
                 f"(gap brightness: r1={s1:.0f}, r2={s2:.0f})")

    result_cv = None

    # ══ Strategy A: Stable Diffusion (GPU) ═══════════════════════════════════
    if DIFFUSION_AVAILABLE:
        notes.append("🤖 Stable Diffusion inpainting …")
        try:
            tgt_pil  = to_pil(target_cv)
            mask_pil = Image.fromarray(mask).convert("L")
            res = pipe(
                prompt=("single natural tooth filling gap in dental model, "
                        "smooth white enamel surface, same size as adjacent teeth, "
                        "matching colour and shape perfectly, photorealistic"),
                negative_prompt=("gap, black hole, dark void, floating, disconnected, "
                                 "blurry, deformed, extra teeth, wrong colour, shadow"),
                image=tgt_pil.resize((512, 512)),
                mask_image=mask_pil.resize((512, 512)),
                num_inference_steps=45,
                guidance_scale=9.0,
            ).images[0].resize(tgt_pil.size, Image.LANCZOS)
            result_cv = to_cv(res)
            notes.append("✨ Diffusion complete.")
        except Exception as e:
            notes.append(f"⚠️  Diffusion failed ({e}) — classical fallback.")
            result_cv = None

    # ══ Strategy B: Symmetry patch from reference ════════════════════════════
    if result_cv is None:
        notes.append("🔄 Symmetry-based fill from reference …")
        try:
            sym = get_symmetry_patch(best_ref, mask, notes)
            if sym:
                patch, cmask, px, py = sym
                patch = colour_match_lab(patch, target_cv, mask)
                result_cv = seamless_paste(target_cv.copy(), patch,
                                           cmask, px, py, notes)
                notes.append("✅ Symmetry fill done.")
        except Exception as e:
            notes.append(f"⚠️  Symmetry fill failed ({e}).")
            result_cv = None

    # ══ Strategy C: Direct reference copy at gap location ════════════════════
    if result_cv is None:
        notes.append("📋 Direct reference copy at gap location …")
        try:
            ref_data = get_ref_patch(best_ref, mask)
            if ref_data:
                patch, cmask, px, py = ref_data
                patch = colour_match_lab(patch, target_cv, mask)
                result_cv = seamless_paste(target_cv.copy(), patch,
                                           cmask, px, py, notes)
                notes.append("✅ Reference copy done.")
        except Exception as e:
            notes.append(f"⚠️  Reference copy failed ({e}).")
            result_cv = None

    # ══ Strategy D: Pure OpenCV inpaint ══════════════════════════════════════
    if result_cv is None:
        result_cv = opencv_inpaint_fill(target_cv, mask, notes)

    # ── Seam refinement ───────────────────────────────────────────────────────
    result_cv = refine_seam(result_cv, mask)
    notes.append("🎨 Seam refinement done.")

    # ── Post-process ──────────────────────────────────────────────────────────
    result_pil = to_pil(result_cv)
    result_pil = result_pil.filter(
        ImageFilter.UnsharpMask(radius=0.9, percent=50, threshold=3))
    result_pil = ImageEnhance.Contrast(result_pil).enhance(1.04)

    final = result_pil.resize(orig_size, Image.LANCZOS)
    notes.append(f"📦 Output: {orig_size[0]}×{orig_size[1]} px")
    notes.append("─" * 50)
    notes.append("ℹ️  Research prototype. Not for clinical use.")

    return np.array(final), "\n".join(notes)


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;500;600&display=swap');
:root{
  --bg:#F0EDE8; --white:#FFFFFF; --cream:#E5E1D8;
  --navy:#1C2B3A; --teal:#2A7F74; --gold:#B8934A;
  --text:#2C2C2C; --muted:#7A7A7A; --r:14px;
  --sh:0 2px 18px rgba(0,0,0,.07);
}
body,.gradio-container{
  background:var(--bg)!important;
  font-family:'Inter',sans-serif!important;
  color:var(--text)!important;
}
.hdr{
  background:linear-gradient(135deg,#1C2B3A 0%,#2C4A60 100%);
  border-radius:var(--r);padding:34px 44px;margin-bottom:20px;
  position:relative;overflow:hidden;
}
.hdr::after{content:'';position:absolute;bottom:-50px;right:-50px;
  width:200px;height:200px;border-radius:50%;
  background:radial-gradient(circle,rgba(42,127,116,.22),transparent 70%);}
.hdr h1{font-family:'Playfair Display',serif!important;font-size:1.85rem!important;
  color:#F0EDE8!important;margin:0 0 10px!important;font-weight:700;}
.hdr p{color:rgba(240,237,232,.72)!important;font-size:.88rem!important;
  margin:0!important;line-height:1.65;max-width:600px;}
.pill{display:inline-block;background:rgba(184,147,74,.18);color:#D4A85A;
  border:1px solid rgba(184,147,74,.35);border-radius:20px;
  padding:3px 14px;font-size:.70rem;font-weight:600;letter-spacing:1px;
  text-transform:uppercase;margin-bottom:14px;}
.card{background:var(--white);border:1px solid var(--cream);
  border-radius:var(--r);padding:16px 20px;box-shadow:var(--sh);margin-bottom:10px;}
.card h3{font-family:'Playfair Display',serif;font-size:.98rem;
  color:var(--navy);margin:0 0 4px;font-weight:500;}
.card p{font-size:.79rem;color:var(--muted);margin:0;line-height:1.55;}
.tip{background:#EEF8F7;border-left:4px solid var(--teal);
  border-radius:0 8px 8px 0;padding:10px 16px;
  font-size:.79rem;color:#1A5048;margin:10px 0;line-height:1.6;}
.go-btn{
  background:linear-gradient(135deg,var(--teal),#1f6059)!important;
  color:#fff!important;border:none!important;border-radius:10px!important;
  font-size:.95rem!important;font-weight:600!important;
  cursor:pointer!important;transition:all .2s!important;
  box-shadow:0 4px 14px rgba(42,127,116,.30)!important;
}
.go-btn:hover{opacity:.87!important;transform:translateY(-2px)!important;}
.notes textarea{font-size:.79rem!important;background:#F7F5F0!important;
  border:1px solid var(--cream)!important;border-radius:8px!important;
  color:#444!important;line-height:1.8!important;}
.warn{background:#FFFBF0;border-left:4px solid var(--gold);
  border-radius:0 8px 8px 0;padding:10px 16px;
  font-size:.78rem;color:#6B4C0E;margin-top:10px;line-height:1.5;}
"""

HEADER = """
<div class="hdr">
  <div class="pill">🦷 Dental Reconstruction v4</div>
  <h1>Dental Surface Reconstruction</h1>
  <p>Upload healthy reference images, then paint over the missing tooth gap
     on the target image. The system reconstructs the tooth using reference
     texture, colour-matching, and seamless blending.</p>
</div>"""

TIP = """<div class="tip">
  <strong>🖌️ How to mark the gap:</strong>
  After uploading the target image below, use the <strong>brush/pen tool</strong>
  in the editor to paint a coloured stroke directly over the missing tooth area.
  Any colour works — just cover the gap fully. Then press <strong>Run Reconstruction</strong>.
</div>"""

WARN = """<div class="warn">
  ⚠️ <strong>Research &amp; Demo Use Only.</strong>
  Not a clinical diagnostic instrument. Always consult a dental professional.
</div>"""

with gr.Blocks(css=CSS, title="Dental Reconstruction v4") as demo:
    gr.HTML(HEADER)

    with gr.Row(equal_height=False):

        # ── LEFT COLUMN ────────────────────────────────────────────────────
        with gr.Column(scale=5):

            gr.HTML("""<div class="card">
              <h3>Step 1 — Healthy Reference Images</h3>
              <p>Upload 1–2 photos of the <strong>complete dental model</strong>
                 (all teeth present). These provide the tooth texture for fill.</p>
            </div>""")
            with gr.Row():
                ref1 = gr.Image(label="Reference — Healthy View 1",
                                type="numpy", height=195)
                ref2 = gr.Image(label="Reference — Healthy View 2",
                                type="numpy", height=195)

            gr.HTML("""<div class="card" style="margin-top:6px">
              <h3>Step 2 — Target: Paint the Gap</h3>
              <p>Upload the target image with the missing tooth, then
                 <strong>paint over the gap</strong> using the brush tool.</p>
            </div>""")
            gr.HTML(TIP)

            target_editor = gr.ImageEditor(
                label="Target Image — Paint Over the Missing Tooth Gap",
                type="numpy",
                height=320,
                brush=gr.Brush(colors=["#FF0000", "#00FF00", "#0000FF"],
                               default_color="#FF0000",
                               default_size=18),
            )

            run_btn = gr.Button("⚙️  Run Reconstruction",
                                elem_classes="go-btn", size="lg")
            gr.HTML(WARN)

        # ── RIGHT COLUMN ───────────────────────────────────────────────────
        with gr.Column(scale=5):

            gr.HTML("""<div class="card">
              <h3>Reconstructed Output</h3>
              <p>The filled result will appear here after you run the pipeline.</p>
            </div>""")

            out_img = gr.Image(label="Reconstructed Dental Surface",
                               type="numpy", height=390, interactive=False)

            out_log = gr.Textbox(
                label="Processing Log", lines=15,
                interactive=False, elem_classes="notes",
                placeholder="Pipeline log appears here after running …")

    run_btn.click(
        fn=reconstruct,
        inputs=[ref1, ref2, target_editor],
        outputs=[out_img, out_log])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
