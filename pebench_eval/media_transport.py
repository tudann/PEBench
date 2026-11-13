from __future__ import annotations

import io
from pathlib import Path
from typing import Any

MAX_INLINE_IMAGE_BYTES = 1280 * 1024
MAX_IMAGE_EDGE = 4096
IMAGE_TRANSPORT_VERSION = "inline-image-1280k-edge4096-v3"


def prepare_image(asset: Any) -> tuple[bytes, str, dict[str, Any]]:
    """Read an image without modifying its source; shrink only when inline payload limits require it."""
    raw = asset.path.read_bytes()
    suffix = asset.path.suffix.lower()
    default_mime = "image/png" if suffix == ".png" else "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/webp"
    if len(raw) <= MAX_INLINE_IMAGE_BYTES:
        return raw, default_mime, {"asset_id": asset.asset_id, "mode": "original", "bytes": len(raw)}
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ValueError(f"{asset.relative_path} exceeds the inline image limit; install Pillow or configure an image URL transport") from None
    try:
        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            if has_alpha and image.convert("RGBA").getchannel("A").getextrema() == (255, 255):
                has_alpha = False
            if has_alpha:
                for edge in (MAX_IMAGE_EDGE, 3584, 3072, 2560, 2048, 1792, 1536, 1280, 1024):
                    scale = min(1.0, edge / max(width, height))
                    current = ImageOps.exif_transpose(source)
                    if scale < 1:
                        current = current.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
                    out = io.BytesIO(); current.save(out, format="PNG", optimize=True)
                    data = out.getvalue()
                    if len(data) <= MAX_INLINE_IMAGE_BYTES:
                        return data, "image/png", {"asset_id": asset.asset_id, "mode": "png", "bytes": len(data),
                                                   "original_bytes": len(raw), "original_width": width, "original_height": height,
                                                   "max_edge": edge, "image_transport_version": IMAGE_TRANSPORT_VERSION}
            else:
                for edge in (MAX_IMAGE_EDGE, 3584, 3072, 2560, 2048, 1792, 1536, 1280, 1024):
                    scale = min(1.0, edge / max(width, height))
                    current = ImageOps.exif_transpose(source).convert("RGB")
                    if scale < 1:
                        current = current.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS)
                    for quality in (92, 88, 84, 80):
                        out = io.BytesIO(); current.save(out, format="JPEG", quality=quality, optimize=True)
                        data = out.getvalue()
                        if len(data) <= MAX_INLINE_IMAGE_BYTES:
                            return data, "image/jpeg", {"asset_id": asset.asset_id, "mode": f"jpeg-q{quality}",
                                                          "bytes": len(data), "original_bytes": len(raw),
                                                          "original_width": width, "original_height": height,
                                                          "max_edge": edge, "image_transport_version": IMAGE_TRANSPORT_VERSION}
            raise ValueError(f"could not reduce {asset.relative_path} below the inline image limit")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not prepare image {asset.relative_path}: {type(exc).__name__}") from None
