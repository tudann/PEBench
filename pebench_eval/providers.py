from __future__ import annotations

import base64
import json
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

import requests

from .dataset import Case
from .evidence import (asset_manifest_evidence, normalize_fact_response,
                       text_only_evidence)
from .prompts import EVIDENCE_SYSTEM, evidence_user, judge_system, judge_user
from .prompt_facts import PROMPT_FACTS_SYSTEM, prompt_facts_user, validate_prompt_facts
from .schema import DIMENSIONS, RUBRIC_VERSION
from .media_transport import IMAGE_TRANSPORT_VERSION, prepare_image


class ProviderError(RuntimeError):
    pass


def salvage_json_object(text: str) -> dict[str, Any] | None:
    """Recover a JSON object whose tail was cut off by the output token limit."""
    start = text.find("{")
    if start < 0:
        return None
    source = text[start:]
    in_string = False
    escaped = False
    stack: list[str] = []
    for char in source:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    if not stack and not in_string:
        return None
    closed = source
    if in_string:
        closed += '"'
    closed = closed.rstrip()
    if closed.endswith(","):
        closed = closed[:-1]
    for opener in reversed(stack):
        closed += "}" if opener == "{" else "]"
    try:
        value = json.loads(closed)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


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


    def plan_text(self, sample):
        from .text_rubric import ALL_CODES, DIMENSIONS
        return {"requirements": [{"id": "R1", "criterion": "SAP", "source_quote": sample["prompt"],
                                  "statement": "Mock fixture only", "importance": "normal"}],
                "applicability": {c: {"applicable": c == "SAP" or c in DIMENSIONS["IC"][1] or c in DIMENSIONS["DC"][1],
                                      "reason": "Mock protocol fixture, not a semantic decision"} for c in ALL_CODES}}

    def judge_text_pair(self, sample, candidates, plan, dimension):
        from .text_rubric import DIMENSIONS
        answers = {}
        for label, candidate in zip(("A", "B"), candidates):
            if dimension == "SF":
                answer = {"requirements": [{"id": r["id"], "status": "preserved", "pe_quote": candidate["pe_prompt"],
                                            "reason": "Mock fixture only", "impact": ""} for r in plan["requirements"]]}
            else:
                rows = []
                for c in DIMENSIONS[dimension][1]:
                    applies = plan["applicability"][c]["applicable"]
                    row = {"criterion": c, "pe_quotes": [candidate["pe_prompt"]] if applies else [],
                           "rationale": "Mock fixture only"}
                    if dimension == "IC":
                        row.update(severity="none" if applies else None, issue_type="none", issue="", impact="")
                    else:
                        row.update(rating=3 if applies else None, enhancement_type="restatement" if applies else "not_applicable", source_gap="")
                    rows.append(row)
                answer = {"criteria": rows}
            answers[label] = answer
        return {"dimension": dimension, "summary": "MOCK ONLY", "candidates": answers}


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
    _rate_lock: Any = field(default_factory=threading.Lock, init=False, repr=False)
    _request_local: Any = field(default_factory=threading.local, init=False, repr=False)

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
                with self._rate_lock:
                    now = time.monotonic()
                    request_at = max(now, self._next_request_at)
                    self._next_request_at = request_at + 60.0 / self.requests_per_minute
                if request_at > now:
                    time.sleep(request_at - now)
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
                raw_response = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                usage = envelope.get("usage") if isinstance(envelope.get("usage"), dict) else None
                self._request_local.raw_response = raw_response
                self._request_local.usage = usage
                self.last_raw_response = raw_response
                self.last_usage = usage
                response_text = raw_response.strip()
                if response_text.startswith("```"):
                    lines = response_text.splitlines()
                    if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
                        raise ValueError("malformed Markdown JSON fence")
                    response_text = "\n".join(lines[1:-1]).strip()
                try:
                    value = json.loads(response_text)
                except json.JSONDecodeError:
                    value = salvage_json_object(response_text)
                    if value is None:
                        raise
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

    def extract_prompt_requirements(self, case: Case) -> dict[str, Any]:
        feedback = None
        last_error = None
        for attempt in range(3):
            value = self._request(PROMPT_FACTS_SYSTEM, prompt_facts_user(case, feedback))
            try:
                return validate_prompt_facts(value, case)
            except ValueError as exc:
                last_error = exc
                feedback = f"Your previous JSON failed validation: {exc}. Correct the schema and ensure every quote is copied exactly from original_prompt."
        raise ProviderError(f"source requirement extraction failed validation after 3 attempts: {last_error}")

    def _finish_judgment(self, value: dict[str, Any]) -> dict[str, Any]:
        value.update({"provider": self.identity, "provider_version": self.version,
                      "mock": False, "rubric_version": RUBRIC_VERSION,
                      "_pebench_raw_model_response": getattr(self._request_local, "raw_response", ""),
                      "_pebench_usage": getattr(self._request_local, "usage", None)})
        return value

    def _complete_missing_criteria(self, sample: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any],
                                   dimension: str, partial: dict[str, Any], missing: list[str]) -> dict[str, Any]:
        request = {"dimension": dimension, "missing_criteria": missing,
                   "original_prompt": sample["prompt"], "enhanced_prompt": candidate["pe_prompt"],
                   "duration_seconds": sample["duration"], "evidence_facts": evidence,
                   "instruction": "Return one JSON object whose criteria array contains exactly these missing criterion codes and no others. Each entry needs criterion, applicability, severity, issue, evidence, and rationale."}
        filled = self._request(judge_system(dimension), json.dumps(request, ensure_ascii=False, sort_keys=True))
        rows = list(partial.get("criteria") or [])
        extra = filled.get("criteria") if isinstance(filled, dict) else None
        if isinstance(extra, dict):
            extra = list(extra.values())
        if isinstance(extra, list):
            rows.extend(extra)
        partial["criteria"] = rows
        if not isinstance(partial.get("summary"), str) and isinstance(filled, dict) and isinstance(filled.get("summary"), str):
            partial["summary"] = filled["summary"]
        return partial

    def judge(self, sample: dict[str, Any], candidate: dict[str, Any], evidence: dict[str, Any], dimension: str) -> dict[str, Any]:
        from .prompts import validate_dimension_result
        from .schema import DIMENSIONS
        feedback = None
        last_error: Exception | None = None
        repair = ("Return exactly one JSON object. Every criterion entry must be an object with all required codes. "
                  "issue, rationale, evidence.source and evidence.quote must be strings; use an empty string instead of null. "
                  "evidence must be an array of objects.")
        for _attempt in range(3):
            try:
                value = self._request(judge_system(dimension), judge_user(sample, candidate, evidence, dimension, feedback))
            except ProviderError as exc:
                last_error = exc
                raw = getattr(self._request_local, "raw_response", "") or ""
                salvaged = salvage_json_object(raw) if "invalid JSON" in str(exc) else None
                if salvaged is None:
                    feedback = f"Your previous response was cut off or was not valid JSON. {repair}"
                    continue
                value = salvaged
            try:
                checked = validate_dimension_result(value, dimension)
            except ValueError as exc:
                last_error = exc
                present = {item.get("criterion") for item in value.get("criteria") or [] if isinstance(item, dict)}
                missing = [code for code in DIMENSIONS[dimension][1] if code not in present]
                if missing and len(missing) < len(DIMENSIONS[dimension][1]):
                    try:
                        value = self._complete_missing_criteria(sample, candidate, evidence, dimension, value, missing)
                        checked = validate_dimension_result(value, dimension)
                    except (ProviderError, ValueError) as fill_error:
                        last_error = fill_error
                        feedback = f"Your previous JSON failed validation: {exc}. {repair}"
                        continue
                else:
                    feedback = f"Your previous JSON failed validation: {exc}. {repair}"
                    continue
            return self._finish_judgment(checked)
        raise ProviderError(f"judgment failed validation after 3 attempts: {last_error}")


    def _text_request(self, system, payload, validator, draft=None):
        from .text_rubric import RepairNeeded, REPAIR_SYSTEM, CLASSIFY_SYSTEM, DIMENSIONS, CRITERIA, merge_repairs
        feedback, pending, attempts, last_error = None, None, [], None
        if draft is not None:
            try:
                checked = validator(draft)
                checked.update(_pebench_raw_model_response="", _pebench_usage=None, _pebench_attempts=[])
                return checked
            except RepairNeeded as exc:
                pending, last_error = exc, exc
            except ValueError as exc:
                feedback, last_error = str(exc), exc
        for index in range(3):
            request = json.loads(payload)
            if feedback: request["validation_feedback"] = feedback
            if pending:
                request["repair"] = {"targets": pending.targets}
                # Reassess only invalid rows from evidence, without anchoring on the failed draft.
                if "source_plan" in request and "requirements" in request["source_plan"]:
                    ids = {t["path"][-1] for t in pending.targets if t["path"][0] == "candidates" and t["path"][-2] == "requirements"}
                    if ids:
                        request["source_plan"]["requirements"] = [r for r in request["source_plan"]["requirements"] if r["id"] in ids]
                        request["requested_requirement_ids"] = sorted(ids)
            classify = pending is not None and all(t.get("fields") == ["criterion"] and t["path"][0] == "requirements" for t in pending.targets)
            request_system = system + (REPAIR_SYSTEM if pending else "")
            if classify:
                ids = [t["path"][1] for t in pending.targets]
                request = {"original_prompt": json.loads(payload)["original_prompt"],
                           "allowed_criteria": {c: CRITERIA[c] for c in DIMENSIONS["SF"][1]},
                           "requirements_to_classify": [{k: r[k] for k in ("id", "source_quote", "statement")} for r in pending.draft["requirements"] if r["id"] in ids]}
                request_system = CLASSIFY_SYSTEM
                if feedback: request["validation_feedback"] = feedback
            record = {"attempt": index + 1, "kind": "classification_repair" if classify else "local_repair" if pending else "full_response"}
            try:
                # Reset thread-local audit data so transport failures cannot inherit another request.
                self._request_local.raw_response = ""
                self._request_local.usage = None
                raw = self._request(request_system, json.dumps(request, ensure_ascii=False))
                record.update(raw_model_response=getattr(self._request_local, "raw_response", ""),
                              usage=getattr(self._request_local, "usage", None), response=raw)
                if classify:
                    assignments = raw.get("assignments") if isinstance(raw, dict) else None
                    if not isinstance(assignments, dict) or set(assignments) != set(ids):
                        raise ValueError("classification repair must assign exactly the requested requirement IDs")
                    raw = {"patches": [{"path": ["requirements", rid], "value": {"id": rid, "criterion": code}} for rid, code in assignments.items()]}
                if pending: raw = merge_repairs(pending, raw)
                checked = validator(raw)
                record["status"] = "valid"
                attempts.append(record)
                checked.update(_pebench_raw_model_response=getattr(self._request_local, "raw_response", ""),
                               _pebench_usage=getattr(self._request_local, "usage", None), _pebench_attempts=attempts)
                return checked
            except (ValueError, ProviderError) as exc:
                last_error = exc
                record.update(status="invalid", error=str(exc),
                              raw_model_response=getattr(self._request_local, "raw_response", ""),
                              usage=getattr(self._request_local, "usage", None))
                attempts.append(record)
                if isinstance(exc, RepairNeeded): pending = exc
                feedback = f"Correct these validation errors: {exc}. Use only supplied exact source/candidate evidence."
        failure = ProviderError(f"text-v3 response failed validation after 3 attempts: {last_error}")
        failure.raw_model_response = getattr(self._request_local, "raw_response", "")
        failure.usage = getattr(self._request_local, "usage", None)
        failure.attempts = attempts
        failure.partial_response = pending.draft if pending else None
        failure.repair_targets = pending.targets if pending else None
        raise failure

    def plan_text(self, sample, draft=None):
        from .text_rubric import PLAN_SYSTEM, plan_user, validate_plan
        return self._text_request(PLAN_SYSTEM, plan_user(sample), lambda v: validate_plan(v, sample), draft)

    def judge_text_pair(self, sample, candidates, plan, dimension, draft=None):
        from .text_rubric import pair_system, pair_user, validate_pair
        return self._text_request(pair_system(dimension), pair_user(sample, candidates, plan, dimension),
                                   lambda v: validate_pair(v, dimension, candidates, plan), draft)


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

    def extract_prompt_requirements(self, case: Case) -> dict[str, Any]:
        return self.judge_model.extract_prompt_requirements(case)

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
