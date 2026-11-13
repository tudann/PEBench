from __future__ import annotations

from typing import Any

from .dataset import Case, canonical_json
from .prompts import EVIDENCE_PROMPT_VERSION


def text_only_evidence(case: Case, *, provider: str, provider_version: str) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_PROMPT_VERSION, "provider": provider,
        "provider_version": provider_version, "mode": case.mode, "case_key": case.key,
        "assets": [], "cross_asset_relations": [], "limitations": [],
        "source": "original_instruction_only",
    }


def asset_manifest_evidence(case: Case, asset_manifest: dict[str, dict[str, Any]], *, provider: str,
                            provider_version: str, limitations: list[str] | None = None) -> dict[str, Any]:
    """Safe offline evidence placeholder; it never claims to have viewed pixels."""
    assets = []
    for asset in case.assets:
        info = asset_manifest.get(asset.relative_path, {})
        assets.append({"asset_id": asset.asset_id, "asset_type": asset.kind,
                       "input_index": asset.index, "source_path": asset.relative_path,
                       "sha256": info.get("sha256"), "facts": {}, "certainty": "unknown"})
    return {"schema_version": EVIDENCE_PROMPT_VERSION, "provider": provider,
            "provider_version": provider_version, "mode": case.mode, "case_key": case.key,
            "assets": assets, "cross_asset_relations": [], "limitations": list(limitations or []),
            "source": "asset_manifest_only"}


def normalize_fact_response(raw: dict[str, Any], case: Case,
                            asset_manifest: dict[str, dict[str, Any]], *, provider: str,
                            provider_version: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("assets"), list):
        raise ValueError("fact extractor must return an object with an assets array")
    by_id = {}
    for item in raw["assets"]:
        if not isinstance(item, dict) or not isinstance(item.get("asset_id"), str):
            raise ValueError("fact extractor returned an asset without asset_id")
        if item["asset_id"] in by_id:
            raise ValueError(f"duplicate extracted asset_id: {item['asset_id']}")
        by_id[item["asset_id"]] = item
    expected = {asset.asset_id: asset for asset in case.assets}
    if set(by_id) != set(expected):
        raise ValueError(f"fact extractor asset coverage mismatch: missing={sorted(set(expected)-set(by_id))}, extra={sorted(set(by_id)-set(expected))}")
    assets = []
    for asset in case.assets:
        item = by_id[asset.asset_id]
        if item.get("asset_type") not in {None, asset.kind}:
            raise ValueError(f"asset type mismatch for {asset.asset_id}")
        # Identity and hashes come from the frozen input manifest, never from model claims.
        facts = {k: v for k, v in item.items() if k not in {"asset_id", "asset_type", "input_index", "source_path", "sha256"}}
        assets.append({"asset_id": asset.asset_id, "asset_type": asset.kind,
                       "input_index": asset.index, "source_path": asset.relative_path,
                       "sha256": asset_manifest.get(asset.relative_path, {}).get("sha256"),
                       "facts": facts, "certainty": "model_extracted"})
    limitations = raw.get("limitations", [])
    if isinstance(limitations, str):
        limitations = [limitations] if limitations.strip() else []
    if not isinstance(limitations, list):
        raise ValueError("fact extractor limitations must be a list or string")
    relations = raw.get("cross_asset_relations", [])
    if not isinstance(relations, list):
        raise ValueError("fact extractor cross_asset_relations must be a list")
    return {"schema_version": EVIDENCE_PROMPT_VERSION, "provider": provider,
            "provider_version": provider_version, "mode": case.mode, "case_key": case.key,
            "assets": assets, "cross_asset_relations": relations,
            "limitations": limitations, "source": "model_extracted"}


def validate_evidence(value: dict[str, Any], case: Case, asset_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("case_key") != case.key:
        raise ValueError("evidence case_key does not match case")
    if value.get("schema_version") != EVIDENCE_PROMPT_VERSION:
        raise ValueError("unsupported evidence schema version")
    assets = value.get("assets")
    if not isinstance(assets, list):
        raise ValueError("evidence.assets must be a list")
    expected = {asset.asset_id: asset for asset in case.assets}
    seen: set[str] = set()
    for item in assets:
        if not isinstance(item, dict) or item.get("asset_id") not in expected:
            raise ValueError("evidence contains an unknown asset_id")
        asset_id = item["asset_id"]
        if asset_id in seen:
            raise ValueError(f"duplicate evidence asset: {asset_id}")
        seen.add(asset_id)
        asset = expected[asset_id]
        if item.get("asset_type") != asset.kind or item.get("input_index") != asset.index:
            raise ValueError(f"evidence asset identity mismatch: {asset_id}")
        if item.get("source_path") != asset.relative_path:
            raise ValueError(f"evidence source path mismatch: {asset_id}")
        expected_sha = asset_manifest.get(asset.relative_path, {}).get("sha256")
        if expected_sha and item.get("sha256") not in {expected_sha, None}:
            raise ValueError(f"evidence asset hash mismatch: {asset_id}")
        if not isinstance(item.get("facts"), dict):
            raise ValueError(f"evidence facts must be an object: {asset_id}")
    if seen != set(expected):
        raise ValueError(f"evidence asset coverage mismatch: missing={sorted(set(expected)-seen)}")
    if not isinstance(value.get("cross_asset_relations"), list) or not isinstance(value.get("limitations"), list):
        raise ValueError("evidence relations and limitations must be lists")
    return value


def evidence_cache_key(case: Case, asset_manifest: dict[str, dict[str, Any]], provider_name: str,
                       provider_version: str) -> str:
    import hashlib
    assets = [{"path": asset.relative_path, "type": asset.kind,
               "sha256": asset_manifest.get(asset.relative_path, {}).get("sha256")}
              for asset in case.assets]
    from .media_transport import IMAGE_TRANSPORT_VERSION
    payload = {"case": case.to_dict(), "assets": assets, "provider": provider_name,
               "provider_version": provider_version, "prompt_version": EVIDENCE_PROMPT_VERSION,
               "image_transport_version": IMAGE_TRANSPORT_VERSION}
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()
