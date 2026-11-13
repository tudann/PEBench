from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATASETS = {"t2v": "t2v_500.json", "i2v": "i2v_300.json", "r2v": "r2v_300.json"}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Asset:
    index: int
    modality_index: int
    kind: str
    path: Path
    relative_path: str

    @property
    def asset_id(self) -> str:
        return f"{self.kind}_{self.modality_index}"


@dataclass(frozen=True)
class Case:
    mode: str
    capid: str
    prompt: str
    duration: float
    task_category: str
    task_type: str
    assets: tuple[Asset, ...]
    source_file: str

    @property
    def key(self) -> str:
        return f"{self.mode}:{self.capid}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_key": self.key, "mode": self.mode, "capid": self.capid,
            "prompt": self.prompt, "duration": self.duration,
            "task_category": self.task_category, "task_type": self.task_type,
            "assets": [{"index": a.index, "modality_index": a.modality_index,
                        "asset_id": a.asset_id, "type": a.kind, "path": a.relative_path}
                       for a in self.assets],
            "source_file": self.source_file,
        }


def _path_for_asset(data_root: Path, local_path: str) -> Path:
    if not isinstance(local_path, str) or not local_path.strip():
        raise ValueError("media.local_path must be a non-empty string")
    root = data_root.resolve()
    path = (root / local_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset path escapes data directory: {local_path}")
    return path


def load_cases(data_root: str | Path) -> list[Case]:
    root = Path(data_root).resolve()
    cases: list[Case] = []
    expected_mode_by_file = {file: mode for mode, file in DATASETS.items()}
    for mode, filename in DATASETS.items():
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required benchmark file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(f"{filename} must contain a JSON array")
        seen: set[str] = set()
        for row_number, row in enumerate(raw, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{filename}:{row_number}: expected an object")
            capid = str(row.get("capid", "")).strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", capid):
                raise ValueError(f"{filename}:{row_number}: invalid capid: {capid!r}")
            if not capid or capid in seen:
                raise ValueError(f"{filename}:{row_number}: capid is empty or duplicated: {capid!r}")
            seen.add(capid)
            if row.get("generation_type") != expected_mode_by_file[filename]:
                raise ValueError(f"{filename}:{row_number}: generation_type does not match file")
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"{filename}:{row_number}: prompt is empty")
            try:
                duration = float(row["duration"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"{filename}:{row_number}: invalid duration") from None
            if not (duration > 0):
                raise ValueError(f"{filename}:{row_number}: duration must be positive")
            media = row.get("media", []) or []
            if not isinstance(media, list):
                raise ValueError(f"{filename}:{row_number}: media must be a list")
            assets: list[Asset] = []
            modality_counts = {"image": 0, "video": 0}
            for index, item in enumerate(media, 1):
                if not isinstance(item, dict) or item.get("type") not in modality_counts:
                    raise ValueError(f"{filename}:{row_number}: media entry must be image or video")
                kind = item["type"]
                modality_counts[kind] += 1
                local_path = item.get("local_path")
                asset_path = _path_for_asset(root, local_path)
                assets.append(Asset(index=index, modality_index=modality_counts[kind], kind=kind, path=asset_path,
                                    relative_path=Path(local_path).as_posix()))
            if mode == "t2v" and assets:
                raise ValueError(f"{filename}:{row_number}: T2V case unexpectedly has media")
            if mode == "i2v" and (len(assets) != 1 or assets[0].kind != "image"):
                raise ValueError(f"{filename}:{row_number}: I2V requires exactly one first-frame image")
            if mode == "r2v" and not assets:
                raise ValueError(f"{filename}:{row_number}: R2V requires at least one reference asset")
            cases.append(Case(mode, capid, prompt, duration, str(row.get("task_category", "")),
                              str(row.get("task_type", "")), tuple(assets), filename))
    if len({case.key for case in cases}) != len(cases):
        raise ValueError("case keys are not unique across benchmark files")
    return cases


def validate_assets(cases: list[Case], *, inspect_content: bool = True) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for case in cases:
        for asset in case.assets:
            path = asset.path
            if not path.is_file():
                raise FileNotFoundError(f"{case.key}: referenced asset missing: {asset.relative_path}")
            stat = path.stat()
            info = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": sha256_file(path),
                    "type": asset.kind, "path": asset.relative_path}
            suffix = path.suffix.lower()
            if asset.kind == "image":
                mime = mimetypes.guess_type(path.name)[0] or ""
                if not mime.startswith("image/"):
                    raise ValueError(f"{case.key}: unsupported image extension: {asset.relative_path}")
                if inspect_content:
                    try:
                        from PIL import Image
                        with Image.open(path) as image:
                            image.verify()
                        with Image.open(path) as image:
                            info["width"], info["height"] = image.size
                            info["format"] = image.format
                    except ImportError:
                        with path.open("rb") as stream:
                            magic = stream.read(16)
                        if not (magic.startswith(b"\x89PNG\r\n\x1a\n") or magic.startswith(b"\xff\xd8\xff")
                                or (magic[:4] == b"RIFF" and magic[8:12] == b"WEBP")):
                            raise ValueError(f"{case.key}: invalid image bytes: {asset.relative_path}")
                    except Exception as exc:
                        raise ValueError(f"{case.key}: image cannot be decoded ({asset.relative_path}): {exc}") from None
            else:
                if suffix not in {".mp4", ".mov"}:
                    raise ValueError(f"{case.key}: unsupported video extension: {asset.relative_path}")
                if inspect_content:
                    try:
                        result = subprocess.run(
                            ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height",
                             "-of", "json", str(path)], capture_output=True, text=True, timeout=45, check=True)
                        media_info = json.loads(result.stdout)
                        streams = media_info.get("streams", [])
                        if not any(s.get("codec_type") == "video" for s in streams):
                            raise ValueError("no video stream")
                        info["duration_seconds"] = float(media_info["format"]["duration"])
                        info["streams"] = [{k: s.get(k) for k in ("codec_type", "codec_name", "width", "height")}
                                            for s in streams]
                    except FileNotFoundError:
                        raise RuntimeError("ffprobe is required to validate R2V video assets") from None
                    except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError) as exc:
                        raise ValueError(f"{case.key}: video cannot be probed ({asset.relative_path}): {exc}") from None
            manifest[asset.relative_path] = info
    return manifest


def dataset_fingerprint(data_root: str | Path, asset_manifest: dict[str, dict[str, Any]]) -> str:
    root = Path(data_root).resolve()
    source_hashes = {name: sha256_file(root / name) for name in DATASETS.values()}
    value = {"sources": source_hashes, "assets": asset_manifest}
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def case_fingerprint(case: Case, asset_manifest: dict[str, dict[str, Any]]) -> str:
    assets = [{"type": a.kind, "path": a.relative_path,
               "sha256": asset_manifest.get(a.relative_path, {}).get("sha256")}
              for a in case.assets]
    return sha256_bytes(canonical_json({"case": case.to_dict(), "assets": assets}).encode("utf-8"))
