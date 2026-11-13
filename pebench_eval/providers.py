from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import requests

from .dataset import Case
from .evidence import (asset_manifest_evidence, normalize_fact_response,
                       text_only_evidence)
from .prompts import EVIDENCE_SYSTEM, evidence_user, judge_system, judge_user
from .schema import DIMENSIONS, RUBRIC_VERSION
from .media_transport import IMAGE_TRANSPORT_VERSION, prepare_image


class ProviderError(RuntimeError):
    pass


@dataclass
class MockProvider:
    """Protocol-only provider. Its output must never be treated as a semantic score."""
    name: str = "mock"
    version: str = "0.1"
    is_mock: bool = True
    last_raw_response: str = field(default="", init=False)
    last_usage: dict[str, Any] | None = field(default=None, init=False)

    def evidence_provider(self, case: Case):
        return self

    def judge_provider(self):
        return self

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "version": self.version, "identity": f"{self.name}:{self.version}", "mock": True}

    def extract_evidence(self, case: Case, sample: dict[str, Any], asset_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if case.mode == "t2v":
            value = text_only_evidence(case, provider=self.name, provider_version=self.version)
        else:
            value = asset_manifest_evidence(case, asset_manifest, provider=self.name, provider_version=self.version,
                                            limitations=["mock provider did not inspect image/video content"])
        self.last_raw_response = json.dumps(value, ensure_ascii=False)
        return value

    def judge(self, sample: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any], dimension: str) -> dict[str, Any]:
        criteria = [{"criterion": code, "applicability": "applicable", "severity": "none",
                     "issue": "", "evidence": [{"source": "mock", "quote": "protocol smoke test only"}],
                     "rationale": "Synthetic protocol output; not a semantic judgment."}
                    for code in DIMENSIONS[dimension][1]]
        value = {"dimension": dimension, "dimension_score": 100.0,
                 "summary": "MOCK ONLY: protocol smoke test; not a semantic judgment.",
                 "criteria": criteria, "provider": self.name, "provider_version": self.version,
                 "mock": True, "rubric_version": RUBRIC_VERSION}
        self.last_raw_response = json.dumps(value, ensure_ascii=False)
        return value


@dataclass
class OpenAICompatibleProvider:
    endpoint: str
    api_key: str
    model: str
    timeout_seconds: int = 300
    max_tokens: int = 8192
    requests_per_minute: int = 30
    name: str = "openai-compatible"
    version: str = "http-json-v1"
    media_public_base_url: str | None = None
    video_input_type: str = "video_url"
    trust_env: bool = True
    is_mock: bool = False
    last_raw_response: str = field(default="", init=False)
    last_usage: dict[str, Any] | None = field(default=None, init=False)
    last_asset_transforms: list[dict[str, Any]] = field(default_factory=list, init=False)
    _next_request_at: float = field(default=0.0, init=False, repr=False)

    @property
    def identity(self) -> str:
        # Keys and complete endpoint URLs are intentionally excluded from persisted metadata.
        import hashlib
        endpoint_id = hashlib.sha256(self.endpoint.encode("utf-8")).hexdigest()[:12]
        return f"{self.name}:{self.model}:{self.version}:endpoint-{endpoint_id}:tokens-{self.max_tokens}:rpm-{self.requests_per_minute}"

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "version": self.version,
                "identity": self.identity, "mock": False}

    def _asset_url(self, relative_path: str) -> str:
        if not self.media_public_base_url:
            raise ProviderError("video evidence requires PEBENCH_VIDEO_PUBLIC_BASE_URL or a custom provider adapter")
        base = self.media_public_base_url.rstrip("/") + "/"
        safe_path = "/".join(quote(part, safe="") for part in Path(relative_path).parts)
        return urljoin(base, safe_path)

    def _request(self, system: str, user: str, *, assets: list[Any] | None = None) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        self.last_asset_transforms = []
        for asset in assets or []:
            content.append({"type": "text", "text": f"ASSET {asset.asset_id}; input_index={asset.index}; type={asset.kind}; source={asset.relative_path}"})
            if asset.kind == "image":
                raw, mime, transform = prepare_image(asset)
                self.last_asset_transforms.append(transform)
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}", "detail": "high"}})
            elif asset.kind == "video":
                # A video-capable chat endpoint must accept this provider extension and reach the URL.
                if self.video_input_type != "video_url":
                    raise ProviderError(f"unsupported video input type: {self.video_input_type}")
                content.append({"type": "video_url", "video_url": {"url": self._asset_url(asset.relative_path)}})
        body = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "response_format": {"type": "json_object"}}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        for attempt in range(4):
            if self.requests_per_minute > 0:
                now = time.monotonic()
                if self._next_request_at > now:
                    time.sleep(self._next_request_at - now)
                self._next_request_at = time.monotonic() + 60.0 / self.requests_per_minute
            try:
                if self.trust_env:
                    response = requests.post(self.endpoint, headers=headers, json=body, timeout=self.timeout_seconds)
                else:
                    with requests.Session() as session:
                        session.trust_env = False
                        response = session.post(self.endpoint, headers=headers, json=body, timeout=self.timeout_seconds)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    retry_after = response.headers.get("Retry-After", "")
                    try: delay = min(30.0, max(0.5, float(retry_after)))
                    except ValueError: delay = min(20.0, 1.5 ** attempt)
                    time.sleep(delay)
                    continue
                if response.status_code >= 400:
                    raise ProviderError(f"provider returned HTTP {response.status_code}")
                envelope = response.json()
                result = envelope["choices"][0]["message"]["content"]
                if isinstance(result, list):
                    result = "".join(x.get("text", "") for x in result if isinstance(x, dict))
                self.last_raw_response = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                self.last_usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
                response_text = self.last_raw_response.strip()
                if response_text.startswith("```"):
                    lines = response_text.splitlines()
                    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
                        raise ValueError("malformed Markdown JSON fence")
                    response_text = "\n".join(lines[1:-1]).strip()
                value = json.loads(response_text)
                if not isinstance(value, dict):
                    raise ValueError("model JSON top-level must be an object")
                return value
            except ProviderError:
                raise
            except requests.RequestException as exc:
                if attempt < 3:
                    time.sleep(min(20.0, 1.5 ** attempt))
                    continue
                raise ProviderError(f"network error ({type(exc).__name__})") from None
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise ProviderError(f"invalid JSON response ({type(exc).__name__})") from None
        raise ProviderError("provider request exhausted retries")

    def extract_evidence(self, case: Case, sample: dict[str, Any], asset_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raw = self._request(EVIDENCE_SYSTEM, evidence_user(sample), assets=list(case.assets))
        value = normalize_fact_response(raw, case, asset_manifest, provider=self.identity,
                                        provider_version=self.version)
        value["media_transport"] = {"image_transport_version": IMAGE_TRANSPORT_VERSION,
                                    "assets": self.last_asset_transforms}
        return value

    def judge(self, sample: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any], dimension: str) -> dict[str, Any]:
        value = self._request(judge_system(dimension), judge_user(sample, candidate, evidence, dimension))
        value.update({"provider": self.identity, "provider_version": self.version,
                      "mock": False, "rubric_version": RUBRIC_VERSION})
        return value


@dataclass
class ProviderBundle:
    image_facts: OpenAICompatibleProvider
    video_facts: OpenAICompatibleProvider | None
    judge_model: OpenAICompatibleProvider
    name: str = "pebench-provider-bundle"
    is_mock: bool = False

    def evidence_provider(self, case: Case):
        if any(asset.kind == "video" for asset in case.assets):
            if self.video_facts is None:
                raise ProviderError("R2V case contains video but PEBENCH_VIDEO_FACTS_* is not configured")
            return self.video_facts
        return self.image_facts

    def judge_provider(self):
        return self.judge_model

    @property
    def version(self) -> str:
        video_identity = self.video_facts.identity if self.video_facts else "video-provider-unconfigured"
        return f"{self.image_facts.identity}|{video_identity}|{self.judge_model.identity}"

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "mock": False, "image_facts": self.image_facts.metadata(),
                "video_facts": self.video_facts.metadata() if self.video_facts else None, "judge": self.judge_model.metadata()}

    def extract_evidence(self, case: Case, sample: dict[str, Any], asset_manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if case.mode == "t2v":
            return text_only_evidence(case, provider="none", provider_version="no-visual-evidence")
        return self.evidence_provider(case).extract_evidence(case, sample, asset_manifest)

    def judge(self, sample: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any], dimension: str) -> dict[str, Any]:
        return self.judge_model.judge(sample, candidate, evidence, dimension)


def _client(prefix: str, *, fallback: str | None = None, video: bool = False) -> OpenAICompatibleProvider:
    endpoint = os.getenv(prefix + "_ENDPOINT") or (os.getenv(fallback + "_ENDPOINT") if fallback else None)
    api_key = os.getenv(prefix + "_API_KEY") or (os.getenv(fallback + "_API_KEY") if fallback else None)
    model = os.getenv(prefix + "_MODEL") or (os.getenv(fallback + "_MODEL") if fallback else None)
    if not endpoint or not api_key or not model:
        raise ProviderError(f"real provider requires {prefix}_ENDPOINT, {prefix}_API_KEY and {prefix}_MODEL")
    timeout = int(os.getenv("PEBENCH_PROVIDER_TIMEOUT_SECONDS", "300"))
    max_tokens = int(os.getenv("PEBENCH_PROVIDER_MAX_TOKENS", "8192"))
    rpm = int(os.getenv("PEBENCH_PROVIDER_RPM", "30"))
    if not 1 <= timeout <= 3600 or not 256 <= max_tokens <= 65536 or not 0 <= rpm <= 10000:
        raise ProviderError("provider timeout/max_tokens/RPM is outside supported bounds")
    return OpenAICompatibleProvider(endpoint=endpoint, api_key=api_key, model=model,
                                   timeout_seconds=timeout, max_tokens=max_tokens, requests_per_minute=rpm,
                                   name=prefix.lower(), media_public_base_url=os.getenv("PEBENCH_VIDEO_PUBLIC_BASE_URL") if video else None)


def provider_from_env(name: str):
    if name == "mock":
        return MockProvider()
    if name != "real":
        raise ProviderError(f"unsupported provider: {name}")
    judge = _client("PEBENCH_JUDGE")
    images = _client("PEBENCH_IMAGE_FACTS", fallback="PEBENCH_JUDGE")
    video_prefix = "PEBENCH_VIDEO_FACTS"
    if all(os.getenv(video_prefix + suffix) for suffix in ("_ENDPOINT", "_API_KEY", "_MODEL")):
        videos = _client(video_prefix, video=True)
    else:
        videos = None
    return ProviderBundle(image_facts=images, video_facts=videos, judge_model=judge)
