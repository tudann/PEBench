from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .dataset import Case, load_cases
from .providers import MockProvider
from .report import summarize
from .runner import run_pipeline


class CountingMockProvider(MockProvider):
    def __init__(self):
        super().__init__()
        self.evidence_calls = 0
        self.judge_calls = 0

    def extract_evidence(self, *args, **kwargs):
        self.evidence_calls += 1
        return super().extract_evidence(*args, **kwargs)

    def judge(self, *args, **kwargs):
        self.judge_calls += 1
        return super().judge(*args, **kwargs)


def representative_cases(cases: list[Case]) -> list[Case]:
    selected = []
    selected.append(next(c for c in cases if c.mode == "t2v"))
    selected.append(next(c for c in cases if c.mode == "i2v"))
    r2v = [c for c in cases if c.mode == "r2v"]
    selected.append(next(c for c in r2v if all(a.kind == "image" for a in c.assets)))
    selected.append(next(c for c in r2v if any(a.kind == "image" for a in c.assets) and any(a.kind == "video" for a in c.assets)))
    selected.append(next(c for c in r2v if all(a.kind == "video" for a in c.assets)))
    return selected


def run_self_test(data_root: str | Path) -> dict:
    root = Path(data_root).resolve()
    cases = representative_cases(load_cases(root))
    rows = [{"mode": c.mode, "capid": c.capid, "method": "smoke", "pe_prompt": "TEST FIXTURE ONLY: synthetic enhanced prompt."}
            for c in cases]
    provider = CountingMockProvider()
    temp_root_parent = root / ".local-tests"
    temp_root_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pebench-self-test-", dir=temp_root_parent) as temp:
        temp_root = Path(temp)
        candidates = temp_root / "candidates.jsonl"
        candidates.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        keys = {c.key for c in cases}
        run_dir = run_pipeline(data_root=root, output_root=temp_root / "runs", provider=provider,
                               candidate_file=candidates, run_id="offline-smoke", case_filter_keys=keys,
                               inspect_content=False)
        first_counts = (provider.evidence_calls, provider.judge_calls)
        if provider.judge_calls != len(cases) * 3:
            raise AssertionError(f"expected 3 dimension calls for each selected case, got {provider.judge_calls}")
        summary = summarize(run_dir)
        if summary["summary_status"] != "mock_only":
            raise AssertionError(f"mock run was not clearly marked: {summary['summary_status']}")
        if summary["mock_judgment_rows"] != len(cases) * 3:
            raise AssertionError("mock judgments were not counted for smoke-test visibility")
        for method in summary["methods"].values():
            for dimension in method["dimensions"].values():
                if dimension["mean_score"] is not None or dimension["real_score_count"] != 0:
                    raise AssertionError("mock score leaked into real metrics")
                if any(metric["pass_rate"] is not None for metric in dimension["criteria"].values()):
                    raise AssertionError("mock pass rate leaked into real metrics")
        # Re-running the same immutable run verifies both evidence and judge checkpoints.
        run_pipeline(data_root=root, output_root=temp_root / "runs", provider=provider,
                     candidate_file=candidates, run_id="offline-smoke", case_filter_keys=keys,
                     inspect_content=False)
        if (provider.evidence_calls, provider.judge_calls) != first_counts:
            raise AssertionError("resume did not reuse successful checkpoints")
        records = list((run_dir / "judgments").glob("*/*/*.json"))
        for path in records:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload.get("mock"):
                raise AssertionError("smoke response lost its mock marker")
        return {"status": "passed", "mock_only": True, "cases": len(cases),
                "dimensions_per_case": 3, "judgment_records": len(records),
                "evidence_calls": first_counts[0], "judge_calls": first_counts[1],
                "resume_reused_checkpoints": True}
