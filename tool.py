from PIL import Image

# copied from example notebook, thanks for providing it!
def crop_rgb(img, box, scale=1.0, strip=2000):
    """Crop box=(x0, y0, x1, y1) out of img as RGB, optionally shrunk by `scale`.
    The crop is converted and shrunk `strip` rows at a time, so even a 1-bit image that is
    100,000+ px tall never has to exist at full size as RGB."""
    x0, y0, x1, y1 = box
    out = Image.new("RGB", (round((x1 - x0) * scale), round((y1 - y0) * scale)))
    for ys in range(y0, y1, strip):
        ye = min(ys + strip, y1)
        part = img.crop((x0, ys, x1, ye)).convert("RGB")
        top, bottom = round((ys - y0) * scale), round((ye - y0) * scale)
        if scale != 1.0:
            part = part.resize((out.width, max(1, bottom - top)), Image.Resampling.BOX)
        out.paste(part, (0, top))
    return out