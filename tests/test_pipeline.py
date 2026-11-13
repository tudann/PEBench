from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pebench_eval.dataset import load_cases, sha256_file, validate_assets
from pebench_eval.evidence import normalize_fact_response, validate_evidence
from pebench_eval.prompts import evidence_user, validate_dimension_result
from pebench_eval.providers import MockProvider, OpenAICompatibleProvider, ProviderBundle
from pebench_eval.media_transport import MAX_INLINE_IMAGE_BYTES, IMAGE_TRANSPORT_VERSION, prepare_image
from pebench_eval.report import summarize
from pebench_eval.runner import run_pipeline
from pebench_eval.schema import DIMENSIONS


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_root = Path(__file__).resolve().parents[1] / ".test-tmp"
        cls.temp_root.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.root = Path(self.temp.name) / "data"
        self.root.mkdir()
        (self.root / "asset/i2v").mkdir(parents=True)
        (self.root / "asset/r2v").mkdir(parents=True)
        (self.root / "asset/i2v/001.png").write_bytes(b"test-image")
        (self.root / "asset/r2v/001.png").write_bytes(b"test-image-r2v")
        (self.root / "asset/r2v/001.mp4").write_bytes(b"test-video")
        rows = {
            "t2v_500.json": [{"capid": "001", "prompt": "A cat runs.", "duration": 5, "generation_type": "t2v", "task_category": "Action", "task_type": "Daily Activity"}],
            "i2v_300.json": [{"capid": "001", "prompt": "Animate the first frame.", "duration": 5, "generation_type": "i2v", "task_category": "Action", "task_type": "Daily Activity", "media": [{"type": "image", "local_path": "asset/i2v/001.png"}]}],
            "r2v_300.json": [{"capid": "001", "prompt": "Continue the image using the video.", "duration": 5, "generation_type": "r2v", "task_category": "Action", "task_type": "Daily Activity", "media": [{"type": "image", "local_path": "asset/r2v/001.png"}, {"type": "video", "local_path": "asset/r2v/001.mp4"}]}],
        }
        for name, value in rows.items():
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_loader_uses_mode_qualified_ids_and_modality_asset_ids(self):
        cases = load_cases(self.root)
        self.assertEqual([c.key for c in cases], ["t2v:001", "i2v:001", "r2v:001"])
        self.assertEqual([a.asset_id for a in cases[-1].assets], ["image_1", "video_1"])
        self.assertEqual([a.index for a in cases[-1].assets], [1, 2])

    def test_path_traversal_is_rejected(self):
        path = self.root / "i2v_300.json"
        raw = json.loads(path.read_text())
        raw[0]["media"][0]["local_path"] = "../../../outside.png"
        path.write_text(json.dumps(raw))
        with self.assertRaises(ValueError):
            load_cases(self.root)

    def test_dimension_contract_rejects_missing_criterion(self):
        dim = "SF"
        value = {"dimension": dim, "dimension_score": 50, "summary": "ok", "criteria": [
            {"criterion": code, "applicability": "applicable", "severity": "none", "issue": "", "evidence": [], "rationale": ""}
            for code in DIMENSIONS[dim][1][:-1]]}
        with self.assertRaises(ValueError):
            validate_dimension_result(value, dim)

    def test_evidence_user_escapes_literal_image_token_without_mutating_sample(self):
        sample = {"prompt": "Keep the text <image> visible.", "assets": [{"asset_id": "image_1"}]}
        original = json.loads(json.dumps(sample))

        request_text = evidence_user(sample)

        self.assertIn("[image placeholder]", request_text)
        self.assertNotIn("<image>", request_text)
        self.assertEqual(sample, original)

    def test_pipeline_resume_and_mock_metrics_exclusion(self):
        cases = load_cases(self.root)
        candidate = Path(self.temp.name) / "candidate.jsonl"
        candidate.write_text("\n".join(json.dumps({"mode": c.mode, "capid": c.capid, "method": "test", "pe_prompt": "A test prompt."}) for c in cases), encoding="utf-8")
        provider = MockProvider()
        evidence_calls, judge_calls = 0, 0
        original_evidence, original_judge = provider.extract_evidence, provider.judge
        def count_evidence(*args, **kwargs):
            nonlocal evidence_calls
            evidence_calls += 1
            return original_evidence(*args, **kwargs)
        def count_judge(*args, **kwargs):
            nonlocal judge_calls
            judge_calls += 1
            return original_judge(*args, **kwargs)
        provider.extract_evidence = count_evidence
        provider.judge = count_judge
        out = Path(self.temp.name) / "runs"
        run_dir = run_pipeline(data_root=self.root, output_root=out, provider=provider,
                               candidate_file=candidate, run_id="test", inspect_content=False)
        self.assertEqual(judge_calls, 9)
        summary = summarize(run_dir)
        self.assertEqual(summary["summary_status"], "mock_only")
        for method in summary["methods"].values():
            for dimension in method["dimensions"].values():
                self.assertIsNone(dimension["mean_score"])
                self.assertTrue(all(x["pass_rate"] is None for x in dimension["criteria"].values()))
        run_pipeline(data_root=self.root, output_root=out, provider=provider,
                     candidate_file=candidate, run_id="test", inspect_content=False)
        self.assertEqual(judge_calls, 9)
        self.assertEqual(evidence_calls, 3)
        self.assertIn("mode:i2v", summary["methods"]["test"]["groups"])
        self.assertIn("task_type:Daily Activity", summary["methods"]["test"]["groups"])

    def test_prepare_without_candidate_does_not_create_judgments(self):
        cases = load_cases(self.root)
        run_dir = run_pipeline(data_root=self.root, output_root=Path(self.temp.name) / "runs",
                               provider=MockProvider(), run_id="prepare", inspect_content=False)
        self.assertEqual(list((run_dir / "judgments").rglob("*.json")), [])
        self.assertTrue((run_dir / "evidence" / "t2v__001.json").is_file())
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "prepared")

    def test_failed_dimension_isolated_then_resumed(self):
        cases = load_cases(self.root)
        candidate = Path(self.temp.name) / "candidate-retry.jsonl"
        candidate.write_text("\n".join(json.dumps({"mode": c.mode, "capid": c.capid, "method": "retry", "pe_prompt": "A test prompt."}) for c in cases), encoding="utf-8")

        class FailOnceProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.failed = False
                self.judge_calls = 0
            def judge(self, sample, candidate, evidence, dimension):
                self.judge_calls += 1
                if dimension == "SF" and not self.failed:
                    self.failed = True
                    raise RuntimeError("synthetic one-time failure")
                return super().judge(sample, candidate, evidence, dimension)

        provider = FailOnceProvider()
        out = Path(self.temp.name) / "retry-runs"
        run_dir = run_pipeline(data_root=self.root, output_root=out, provider=provider,
                               candidate_file=candidate, run_id="retry", inspect_content=False)
        first = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(first["status"], "completed_with_errors")
        self.assertTrue(list(run_dir.rglob("*.error.json")))
        run_pipeline(data_root=self.root, output_root=out, provider=provider,
                     candidate_file=candidate, run_id="retry", inspect_content=False)
        second = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(second["status"], "completed")
        self.assertFalse(list(run_dir.rglob("*.error.json")))
        self.assertEqual(provider.judge_calls, 10)
        self.assertEqual(summarize(run_dir)["judgment_rows"], 9)

    def test_candidate_run_only_prepares_candidate_cases(self):
        candidate = Path(self.temp.name) / "one-candidate.jsonl"
        candidate.write_text(json.dumps({"mode": "i2v", "capid": "001", "method": "partial", "pe_prompt": "A test prompt."}), encoding="utf-8")
        run_dir = run_pipeline(data_root=self.root, output_root=Path(self.temp.name) / "candidate-only",
                               provider=MockProvider(), candidate_file=candidate, run_id="one",
                               inspect_content=False)
        run = json.loads((run_dir / "run.json").read_text())
        self.assertEqual(run["case_count"], 1)
        self.assertTrue((run_dir / "evidence" / "i2v__001.json").is_file())
        self.assertFalse((run_dir / "evidence" / "t2v__001.json").exists())
        self.assertFalse((run_dir / "evidence" / "r2v__001.json").exists())

    def test_output_root_on_system_disk_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "output_root must be on the configured data disk"):
            run_pipeline(data_root=self.root, output_root="/root/pebench-cache",
                         provider=MockProvider(), run_id="must-not-write", inspect_content=False)

    def test_changed_candidate_content_requires_new_run_id(self):
        candidate = Path(self.temp.name) / "changing.jsonl"
        row = {"mode": "t2v", "capid": "001", "method": "version-a", "pe_prompt": "First version."}
        candidate.write_text(json.dumps(row), encoding="utf-8")
        out = Path(self.temp.name) / "fingerprint-runs"
        run_pipeline(data_root=self.root, output_root=out, provider=MockProvider(),
                     candidate_file=candidate, run_id="stable", case_filter_keys={"t2v:001"},
                     inspect_content=False)
        row["pe_prompt"] = "Second version."
        candidate.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "candidate content changed"):
            run_pipeline(data_root=self.root, output_root=out, provider=MockProvider(),
                         candidate_file=candidate, run_id="stable", case_filter_keys={"t2v:001"},
                         inspect_content=False)

    def test_changed_case_scope_rejected(self):
        out = Path(self.temp.name) / "scope"
        run_pipeline(data_root=self.root, output_root=out, provider=MockProvider(), run_id="scope", case_filter_keys={"t2v:001"}, inspect_content=False)
        with self.assertRaisesRegex(ValueError, "case scope"):
            run_pipeline(data_root=self.root, output_root=out, provider=MockProvider(), run_id="scope", case_filter_keys={"i2v:001"}, inspect_content=False)

    def test_video_provider_missing_only_marks_affected_case(self):
        cases = load_cases(self.root)
        candidate = Path(self.temp.name) / "candidate-missing-video.jsonl"
        candidate.write_text("\n".join(json.dumps({"mode": c.mode, "capid": c.capid, "method": "partial", "pe_prompt": "A test prompt."}) for c in cases), encoding="utf-8")
        provider = ProviderBundle(image_facts=MockProvider(name="image-mock"), video_facts=None,
                                  judge_model=MockProvider(name="judge-mock"))
        run_dir = run_pipeline(data_root=self.root, output_root=Path(self.temp.name) / "partial-runs",
                               provider=provider, candidate_file=candidate, run_id="missing-video",
                               inspect_content=False)
        summary = summarize(run_dir)
        self.assertEqual(json.loads((run_dir / "run.json").read_text())["status"], "completed_with_errors")
        self.assertEqual(summary["judgment_rows"], 6)
        self.assertEqual(summary["error_files"], 1)

    def test_large_image_transport_is_bounded_and_source_is_unchanged(self):
        from PIL import Image
        image_path = Path(self.temp.name) / "large.png"
        Image.effect_noise((2200, 2200), 100).convert("RGB").save(image_path, format="PNG", compress_level=0)
        self.assertGreater(image_path.stat().st_size, MAX_INLINE_IMAGE_BYTES)
        source_hash = sha256_file(image_path)
        payload, mime, detail = prepare_image(SimpleNamespace(asset_id="image_1", path=image_path, relative_path="large.png"))
        self.assertLessEqual(len(payload), MAX_INLINE_IMAGE_BYTES)
        self.assertIn(mime, {"image/png", "image/jpeg"})
        self.assertEqual(detail["image_transport_version"], IMAGE_TRANSPORT_VERSION)
        self.assertEqual(sha256_file(image_path), source_hash)

    def test_large_opaque_alpha_png_falls_back_to_jpeg(self):
        from PIL import Image
        image_path = Path(self.temp.name) / "opaque-alpha.png"
        image = Image.effect_noise((2200, 2200), 100).convert("RGBA")
        image.putalpha(255)
        image.save(image_path, format="PNG", compress_level=0)
        self.assertGreater(image_path.stat().st_size, MAX_INLINE_IMAGE_BYTES)
        source_hash = sha256_file(image_path)
        payload, mime, detail = prepare_image(SimpleNamespace(asset_id="image_1", path=image_path, relative_path="opaque-alpha.png"))
        self.assertLessEqual(len(payload), MAX_INLINE_IMAGE_BYTES)
        self.assertEqual(mime, "image/jpeg")
        self.assertTrue(detail["mode"].startswith("jpeg-q"))
        self.assertEqual(sha256_file(image_path), source_hash)

    @patch("pebench_eval.providers.requests.post")
    def test_provider_builds_ordered_video_parts(self, post):
        class Response:
            status_code = 200
            headers = {}
            def raise_for_status(self): pass
            def json(self): return {"choices": [{"message": {"content": '{"ok": true}'}}]}
        post.return_value = Response()
        case = load_cases(self.root)[-1]
        provider = OpenAICompatibleProvider(endpoint="https://example.invalid/v1/chat/completions", api_key="hidden",
                                            model="test", media_public_base_url="https://assets.example/data/")
        value = provider._request("sys", "user", assets=list(case.assets))
        self.assertTrue(value["ok"])
        content = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertEqual([part["type"] for part in content if part["type"] in {"image_url", "video_url"}], ["image_url", "video_url"])
        video = next(part for part in content if part["type"] == "video_url")
        self.assertEqual(video["video_url"]["url"], "https://assets.example/data/asset/r2v/001.mp4")


    @patch("pebench_eval.providers.requests.post")
    def test_provider_accepts_markdown_fenced_json(self, post):
        class Response:
            status_code = 200
            headers = {}
            def json(self): return {"choices": [{"message": {"content": "```json\n{\"ok\": true}\n```"}}]}
        post.return_value = Response()
        provider = OpenAICompatibleProvider(endpoint="https://example.invalid/v1/chat/completions",
                                            api_key="hidden", model="test")
        self.assertEqual(provider._request("sys", "user"), {"ok": True})
        self.assertIn("```json", provider.last_raw_response)

    def test_fact_response_normalizes_string_limitations(self):
        case = load_cases(self.root)[-1]
        manifest = validate_assets([case], inspect_content=False)
        raw = {"assets": [{"asset_id": asset.asset_id, "asset_type": asset.kind}
                          for asset in case.assets],
               "limitations": "The audio track was not inspected."}
        result = normalize_fact_response(raw, case, manifest, provider="test", provider_version="1")
        self.assertEqual(result["limitations"], ["The audio track was not inspected."])

if __name__ == "__main__":
    unittest.main()
