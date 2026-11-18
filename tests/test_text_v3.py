from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pebench_eval.providers import MockProvider, OpenAICompatibleProvider
from pebench_eval.runner import run_pipeline
from pebench_eval.report import summarize, render_markdown
from pebench_eval.text_rubric import (SCORING, VERSION, overall_score, pair_user, score_assessment,
                                      validate_pair, validate_plan)

PROJECT = Path(__file__).resolve().parents[1]


class TextV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=PROJECT / '.test-tmp')
        self.root = Path(self.temp.name)
        self.sample = {"case_key": "t2v:one", "prompt": "A woman wears a red coat. No music.",
                       "duration": None, "task_type": "Daily Activity"}
        self.candidates = [{"method": "secret-official", "pe_prompt": "A woman wears a red coat. No music."},
                           {"method": "secret-online", "pe_prompt": "A woman wears a red coat. No music."}]
        self.mock = MockProvider()
        self.plan = validate_plan(self.mock.plan_text(self.sample), self.sample)

    def tearDown(self):
        self.temp.cleanup()

    def test_source_plan_uses_exact_quotes_and_shared_applicability(self):
        plan = copy.deepcopy(self.plan)
        plan['requirements'][0]['source_quote'] = 'blue coat'
        with self.assertRaisesRegex(ValueError, 'exact source substring'):
            validate_plan(plan, self.sample)
        plan = copy.deepcopy(self.plan)
        plan['applicability']['SAP']['applicable'] = False
        self.assertTrue(validate_plan(plan, self.sample)['applicability']['SAP']['applicable'])

    def test_candidate_payload_does_not_disclose_method_names(self):
        payload = pair_user(self.sample, self.candidates, self.plan, 'DC')
        self.assertNotIn('secret-official', payload)
        self.assertNotIn('secret-online', payload)
        self.assertNotIn('assets', json.loads(payload))

    def test_ic_payload_excludes_source_requirements_to_prevent_double_penalty(self):
        payload = json.loads(pair_user(self.sample, self.candidates, self.plan, 'IC'))
        self.assertNotIn('original_prompt', payload)
        self.assertNotIn('requirements', payload['source_plan'])
        self.assertEqual(payload['source_plan']['applicability'], self.plan['applicability'])
        sf = json.loads(pair_user(self.sample, self.candidates, self.plan, 'SF'))
        self.assertEqual(sf['original_prompt'], self.sample['prompt'])

    def test_fidelity_weights_source_requirements_and_critical_cap(self):
        self.plan['requirements'][0]['importance'] = 'critical'
        intact = {'requirements': [{'id': 'R1', 'status': 'preserved'}]}
        weaker = {'requirements': [{'id': 'R1', 'status': 'weakened'}]}
        broken = {'requirements': [{'id': 'R1', 'status': 'contradicted'}]}
        self.assertEqual(score_assessment(intact, 'SF', self.plan)['dimension_score'], 100)
        self.assertEqual(score_assessment(weaker, 'SF', self.plan)['dimension_score'], 75)
        bad = score_assessment(broken, 'SF', self.plan)
        self.assertEqual(bad['dimension_score'], 0)
        dims = {'SF': bad, 'IC': {'dimension_score': 100, 'critical_violation': False},
                'DC': {'dimension_score': 100, 'critical_violation': False}}
        total = overall_score(dims)
        self.assertEqual(total['uncapped_score'], 80)
        self.assertEqual(total['score'], 69)

    def test_positive_dc_quality_is_distinct_from_error_severity(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        answer = value['candidates']['A']
        self.assertEqual(score_assessment(answer, 'DC', self.plan)['dimension_score'], 75)
        answer['criteria'][0]['rating'] = 4
        self.assertGreater(score_assessment(answer, 'DC', self.plan)['dimension_score'], 75)
        value['candidates']['B'] = copy.deepcopy(answer)
        answer['criteria'][0]['enhancement_type'] = 'padding'
        with self.assertRaisesRegex(ValueError, 'padding/conflict'):
            validate_pair(value, 'DC', self.candidates, self.plan)

    def test_identical_texts_require_same_decisions_and_quotes_cannot_be_invented(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        value['candidates']['B']['criteria'][0]['rating'] = 2
        with self.assertRaisesRegex(ValueError, 'identical candidate'):
            validate_pair(value, 'DC', self.candidates, self.plan)
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        value['candidates']['A']['criteria'][0]['pe_quotes'] = ['A blue coat']
        with self.assertRaisesRegex(ValueError, 'exact nonempty contiguous'):
            validate_pair(value, 'DC', self.candidates, self.plan)

    def test_preserved_requirement_can_omit_empty_impact_but_violation_cannot(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'SF')
        for label in ('A', 'B'):
            del value['candidates'][label]['requirements'][0]['impact']
        self.assertEqual(validate_pair(value, 'SF', self.candidates, self.plan)['candidates']['A']['requirements'][0]['impact'], '')
        value['candidates']['A']['requirements'][0]['status'] = 'contradicted'
        with self.assertRaisesRegex(ValueError, 'concrete impact'):
            validate_pair(value, 'SF', self.candidates, self.plan)

    def test_internal_conflict_needs_two_distinct_candidate_passages(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'IC')
        row = value['candidates']['A']['criteria'][0]
        row.update(severity='minor', issue_type='identity_drift', issue='Unproven identity drift', impact='Changes subject')
        with self.assertRaisesRegex(ValueError, 'two distinct exact quotes'):
            validate_pair(value, 'IC', self.candidates, self.plan)

    def _dataset(self):
        data = self.root / 'data'
        data.mkdir()
        (data / 't2v_500.json').write_text(json.dumps([{'capid': 'one', 'prompt': self.sample['prompt'],
            'duration': None, 'generation_type': 't2v', 'task_category': 'Test', 'task_type': 'Test'}]))
        for name in ['i2v_300.json', 'r2v_300.json']:
            (data / name).write_text('[]')
        path = self.root / 'candidates.jsonl'
        path.write_text('\n'.join(json.dumps({'mode': 't2v', 'capid': 'one', **r}) for r in self.candidates))
        return data, path

    def test_pipeline_order_swap_resume_mock_exclusion_and_historical_guard(self):
        data, candidates = self._dataset()
        class Counting(MockProvider):
            calls = 0
            def judge_text_pair(self, *args):
                self.calls += 1
                return super().judge_text_pair(*args)
        provider = Counting()
        args = dict(data_root=data, output_root=self.root / 'runs', provider=provider, candidate_file=candidates,
                    modes={'t2v'}, run_id='paired', protocol='text-v3')
        run = run_pipeline(**args)
        self.assertEqual(provider.calls, 6)
        summary = summarize(run)
        self.assertEqual(summary['mock_judgment_rows'], 6)
        self.assertEqual(summary['comparison']['mock_pairs'], 1)
        self.assertEqual(summary['comparison']['complete_real_pairs'], 0)
        self.assertIsNone(summary['methods']['secret-official']['overall']['mean_score'])
        self.assertIn('0–4', render_markdown(summary))
        run_pipeline(**args)
        self.assertEqual(provider.calls, 6)
        run_pipeline(**args, resume=False)
        self.assertEqual(provider.calls, 12)
        missing = run / 'judgments/t2v__one.json/secret-online/DC.json'
        missing.unlink()
        run_pipeline(**args)
        self.assertEqual(provider.calls, 12)
        self.assertTrue(missing.is_file())
        meta = json.loads((run / 'run.json').read_text())
        meta['rubric_version'] = 'pebench-24-v2-shootable'
        (run / 'run.json').write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, 'different protocol'):
            run_pipeline(**args)

    def test_order_bias_triggers_review_and_stays_unresolved(self):
        data, candidates = self._dataset()
        self.candidates[1]['pe_prompt'] = 'A woman in a red coat. No music.'
        candidates.write_text('\n'.join(json.dumps({'mode': 't2v', 'capid': 'one', **r}) for r in self.candidates))
        class PositionBiased(MockProvider):
            calls = 0
            def judge_text_pair(self, *args):
                self.calls += 1
                value = super().judge_text_pair(*args)
                if args[-1] == 'DC':
                    for row in value['candidates']['A']['criteria']: row['rating'] = 4
                    for row in value['candidates']['B']['criteria']: row['rating'] = 1
                return value
        provider = PositionBiased()
        run = run_pipeline(data_root=data, output_root=self.root / 'runs', provider=provider,
                           candidate_file=candidates, modes={'t2v'}, run_id='biased', protocol='text-v3')
        self.assertEqual(provider.calls, 12)
        pair = json.loads((run / 'pairs/t2v__one.json').read_text())
        self.assertTrue(pair['reviewed'])
        self.assertEqual(pair['preference'], 'unresolved')

    def test_passage_ids_resolve_verbatim_and_unknown_ids_are_rejected(self):
        from pebench_eval.text_rubric import passages
        pieces = passages(self.candidates[0]['pe_prompt'])
        self.assertEqual(''.join(p['text'] for p in pieces), self.candidates[0]['pe_prompt'])
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'SF')
        for label in ('A', 'B'):
            row = value['candidates'][label]['requirements'][0]
            del row['pe_quote']
            row['pe_ref'] = pieces[0]['id']
        checked = validate_pair(value, 'SF', self.candidates, self.plan)
        self.assertEqual(checked['candidates']['A']['requirements'][0]['pe_quote'], pieces[0]['text'].strip())
        value['candidates']['A']['requirements'][0]['pe_ref'] = 'P999'
        with self.assertRaisesRegex(ValueError, 'unknown candidate passage'):
            validate_pair(value, 'SF', self.candidates, self.plan)

    def test_direct_candidate_arrays_are_normalized_without_relaxing_evidence(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'IC')
        value['candidates'] = {label: answer['criteria'] for label, answer in value['candidates'].items()}
        checked = validate_pair(value, 'IC', self.candidates, self.plan)
        self.assertEqual(len(checked['candidates']['A']['criteria']), 6)
        value['candidates']['A'][0]['pe_quotes'] = ['made-up candidate text']
        with self.assertRaisesRegex(ValueError, 'exact nonempty contiguous'):
            validate_pair(value, 'IC', self.candidates, self.plan)

    def test_provider_retries_invalid_exact_quote_with_feedback(self):
        provider = OpenAICompatibleProvider(endpoint='https://example.invalid', api_key='hidden', model='test')
        valid = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        invalid = copy.deepcopy(valid)
        invalid['candidates']['A']['criteria'][0]['pe_quotes'] = ['fabricated evidence']
        requests = []
        def request(system, user):
            requests.append(json.loads(user))
            return invalid if len(requests) == 1 else {'patches': [{'path': ['candidates', 'A', 'criteria', 'STC'], 'value': valid['candidates']['A']['criteria'][0]}]}
        provider._request = request
        result = provider.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        self.assertEqual(len(requests), 2)
        self.assertIn('validation_feedback', requests[1])
        self.assertEqual(result['dimension'], 'DC')

    def test_plan_repair_changes_only_invalid_classification(self):
        from pebench_eval.text_rubric import RepairNeeded, merge_repairs
        bad = copy.deepcopy(self.plan)
        bad['requirements'][0]['criterion'] = 'VSS'
        with self.assertRaises(RepairNeeded) as caught:
            validate_plan(bad, self.sample)
        error = caught.exception
        self.assertEqual(error.targets[0]['fields'], ['criterion'])
        row = {**bad['requirements'][0], 'criterion': 'GSF'}
        patched = merge_repairs(error, {'patches': [{'path': ['requirements', 'R1'], 'value': row}]})
        self.assertEqual(patched['requirements'][0]['source_quote'], self.sample['prompt'])
        row['importance'] = 'critical'
        with self.assertRaisesRegex(ValueError, 'frozen requirement field'):
            merge_repairs(error, {'patches': [{'path': ['requirements', 'R1'], 'value': row}]})

    def test_sf_applicability_derived_and_plan_payload_separates_enums(self):
        from pebench_eval.text_rubric import plan_user
        payload = json.loads(plan_user(self.sample))
        self.assertNotIn('VSS', payload['requirement_criteria'])
        self.assertIn('VSS', payload['applicability_criteria'])
        draft = copy.deepcopy(self.plan)
        for c in list(draft['applicability']):
            if c in payload['requirement_criteria']: del draft['applicability'][c]
        self.assertTrue(validate_plan(draft, self.sample)['applicability']['SAP']['applicable'])

    def test_frozen_na_is_excluded_even_with_rating_or_invalid_refs(self):
        plan = copy.deepcopy(self.plan)
        plan['applicability']['AVS'] = {'applicable': False, 'reason': 'Source has no audio'}
        plan['applicability']['VSS'] = {'applicable': False, 'reason': 'Source does not request a visual style'}
        for dim, code in [('IC', 'AVS'), ('DC', 'VSS')]:
            value = self.mock.judge_text_pair(self.sample, self.candidates, plan, dim)
            for label in ('A', 'B'):
                row = next(r for r in value['candidates'][label]['criteria'] if r['criterion'] == code)
                row.update(severity='major', rating=4, pe_refs=['P999'], enhancement_type='useful_detail')
            clean = validate_pair(value, dim, self.candidates, plan)
            row = next(r for r in clean['candidates']['A']['criteria'] if r['criterion'] == code)
            self.assertIsNone(row['severity' if dim == 'IC' else 'rating'])
            score = score_assessment(clean['candidates']['A'], dim, plan)['dimension_score']
            self.assertEqual(score, 100 if dim == 'IC' else 75)
            for label in ('A', 'B'):
                value['candidates'][label]['criteria'] = [r for r in value['candidates'][label]['criteria'] if r['criterion'] != code]
            self.assertEqual(validate_pair(value, dim, self.candidates, plan), clean)

    def test_applicable_bad_rating_is_never_filled_by_program(self):
        from pebench_eval.text_rubric import RepairNeeded
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        value['candidates']['A']['criteria'][0]['rating'] = None
        with self.assertRaises(RepairNeeded) as caught:
            validate_pair(value, 'DC', self.candidates, self.plan)
        self.assertEqual(caught.exception.targets[0]['path'], ['candidates', 'A', 'criteria', 'STC'])

    def test_multi_passage_refs_keep_all_evidence_and_reject_unknown_token(self):
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'SF')
        for label in ('A', 'B'):
            value['candidates'][label]['requirements'][0]['pe_refs'] = ['P1', 'P2']
        clean = validate_pair(value, 'SF', self.candidates, self.plan)
        self.assertEqual(len(clean['candidates']['A']['requirements'][0]['pe_quotes']), 2)
        value['candidates']['A']['requirements'][0]['pe_refs'] = 'P1,P999'
        with self.assertRaisesRegex(ValueError, 'unknown candidate passage'):
            validate_pair(value, 'SF', self.candidates, self.plan)

    def test_missing_row_repair_preserves_all_other_decisions(self):
        from pebench_eval.text_rubric import RepairNeeded, merge_repairs
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'SF')
        missing = value['candidates']['A']['requirements'].pop()
        with self.assertRaises(RepairNeeded) as caught:
            validate_pair(value, 'SF', self.candidates, self.plan)
        original_b = copy.deepcopy(caught.exception.draft['candidates']['B'])
        patched = merge_repairs(caught.exception, {'patches': [{'path': ['candidates', 'A', 'requirements', 'R1'], 'value': missing}]})
        self.assertEqual(patched['candidates']['B'], original_b)
        with self.assertRaisesRegex(ValueError, 'unrequested'):
            merge_repairs(caught.exception, {'patches': [{'path': ['candidates', 'B', 'requirements', 'R1'], 'value': missing}]})
        self.assertEqual(validate_pair(patched, 'SF', self.candidates, self.plan)['dimension'], 'SF')

    def test_failed_partial_response_resumes_only_invalid_row(self):
        from pebench_eval.text_runner import _cached_request
        from pebench_eval.providers import ProviderError
        provider = OpenAICompatibleProvider(endpoint='https://example.invalid', api_key='hidden', model='test')
        valid = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'DC')
        bad = copy.deepcopy(valid)
        bad['candidates']['A']['criteria'][0]['rating'] = None
        calls = []
        def fail(system, user):
            calls.append(json.loads(user))
            if 'repair' in calls[-1]: return {'patches': []}
            return bad
        provider._request = fail
        path = self.root / 'cache.json'
        request = lambda draft: provider.judge_text_pair(self.sample, self.candidates, self.plan, 'DC', draft=draft)
        check = lambda v: validate_pair(v, 'DC', self.candidates, self.plan)
        with self.assertRaises(ProviderError): _cached_request(path, 'key', request, check)
        self.assertTrue(path.with_suffix('.pending.json').is_file())
        def repair(system, user):
            calls.append(json.loads(user))
            self.assertIn('repair', calls[-1])
            return {'patches': [{'path': ['candidates', 'A', 'criteria', 'STC'], 'value': valid['candidates']['A']['criteria'][0]}]}
        provider._request = repair
        clean = _cached_request(path, 'key', request, check)
        self.assertEqual(len(calls), 4)
        self.assertEqual(clean['candidates']['B'], check(valid)['candidates']['B'])
        self.assertFalse(path.with_suffix('.pending.json').exists())

    def test_checkpoint_import_preserves_assessments_and_rejects_input_changes(self):
        data, candidates = self._dataset()
        from pebench_eval.text_runner import run_text_pipeline
        class Counting(MockProvider):
            calls = 0
            def judge_text_pair(self, *args):
                self.calls += 1
                return super().judge_text_pair(*args)
        provider = Counting()
        args = dict(data_root=data, output_root=self.root / 'runs', provider=provider, candidate_file=candidates, modes={'t2v'})
        source = run_text_pipeline(**args, run_id='source')
        dest = run_text_pipeline(**args, run_id='imported', checkpoint_source=source)
        self.assertEqual(provider.calls, 6)
        imported = json.loads((dest / 'checkpoint_import.json').read_text())
        self.assertEqual(imported['pairwise'], 6)
        self.assertEqual(imported['plans'], 1)
        merged = run_text_pipeline(**args, run_id='merged', checkpoint_source=[source, dest])
        self.assertEqual(provider.calls, 6)
        self.assertEqual(len(json.loads((merged / 'checkpoint_import.json').read_text())['sources']), 2)
        rows = [json.loads(l) for l in candidates.read_text().splitlines()]
        rows[0]['pe_prompt'] += ' Additional text.'
        candidates.write_text('\n'.join(json.dumps(r) for r in rows))
        with self.assertRaisesRegex(ValueError, 'candidate changed'):
            run_text_pipeline(**args, run_id='changed', checkpoint_source=source)

    def test_classification_repair_hides_other_dimension_codes(self):
        provider = OpenAICompatibleProvider(endpoint='https://example.invalid', api_key='hidden', model='test')
        bad = copy.deepcopy(self.plan)
        bad['requirements'][0]['criterion'] = 'VSS'
        requests = []
        def request(system, user):
            requests.append(json.loads(user))
            self.assertNotIn('applicability', requests[-1])
            self.assertNotIn('VSS', requests[-1]['allowed_criteria'])
            return {'assignments': {'R1': 'GSF'}}
        provider._request = request
        result = provider.plan_text(self.sample, draft=bad)
        self.assertEqual(len(requests), 1)
        self.assertEqual(result['requirements'][0]['criterion'], 'GSF')
        self.assertEqual(result['requirements'][0]['source_quote'], bad['requirements'][0]['source_quote'])
        self.assertEqual(result['requirements'][0]['importance'], bad['requirements'][0]['importance'])

    def test_sf_batches_bound_response_and_resume_completed_parts(self):
        from pebench_eval.text_runner import _sf_batches
        from pebench_eval.providers import ProviderError
        plan = copy.deepcopy(self.plan)
        plan['requirements'] = [{**plan['requirements'][0], 'id': f'R{i}'} for i in range(1, 20)]
        class Interrupted(MockProvider):
            calls = []
            fail = True
            def judge_text_pair(self, sample, candidates, subset, dim):
                self.calls.append([r['id'] for r in subset['requirements']])
                if self.fail and len(self.calls) == 2: raise ProviderError('temporary interruption')
                return super().judge_text_pair(sample, candidates, subset, dim)
        judge = Interrupted()
        args = (self.root / 'SF.forward.json', 'key', self.sample, self.candidates, plan, judge, True)
        with self.assertRaises(ProviderError): _sf_batches(*args)
        judge.fail = False
        result = _sf_batches(*args)
        self.assertEqual(len(judge.calls), 4)
        self.assertTrue(all(len(batch) <= 8 for batch in judge.calls))
        self.assertEqual(len(result['candidates']['A']['requirements']), 19)
        self.assertEqual(score_assessment(result['candidates']['A'], 'SF', plan)['dimension_score'], 100)
        _sf_batches(*args)
        self.assertEqual(len(judge.calls), 4)

    def test_checkpoint_import_reuses_sf_batches_after_parent_interruption(self):
        from pebench_eval.text_runner import run_text_pipeline
        data, candidates = self._dataset()
        class ManyRequirements(MockProvider):
            calls = 0
            def plan_text(self, sample):
                plan = super().plan_text(sample)
                plan['requirements'] = [{**plan['requirements'][0], 'id': f'R{i}'} for i in range(1, 20)]
                return plan
            def judge_text_pair(self, *args):
                self.calls += 1
                return super().judge_text_pair(*args)
        provider = ManyRequirements()
        args = dict(data_root=data, output_root=self.root / 'runs', provider=provider, candidate_file=candidates, modes={'t2v'})
        source = run_text_pipeline(**args, run_id='batches-source')
        self.assertEqual(provider.calls, 10)
        # Simulate interruption between the last successful batch and parent aggregation.
        (source / 'pairwise/t2v__one.json/round-0/SF.forward.json').unlink()
        dest = run_text_pipeline(**args, run_id='batches-dest', checkpoint_source=source)
        self.assertEqual(provider.calls, 10)
        imported = json.loads((dest / 'checkpoint_import.json').read_text())
        self.assertEqual(imported['sf_parts'], 6)
        self.assertTrue((dest / 'pairs/t2v__one.json').is_file())

    def test_nested_candidate_response_keeps_explicit_side_and_repairs_other(self):
        from pebench_eval.text_rubric import RepairNeeded, merge_repairs
        value = self.mock.judge_text_pair(self.sample, self.candidates, self.plan, 'SF')
        only_b = value['candidates']['B']
        value['candidates'] = {'A': {'B': only_b}}
        with self.assertRaises(RepairNeeded) as caught:
            validate_pair(value, 'SF', self.candidates, self.plan)
        targets = caught.exception.targets
        self.assertEqual({t['path'][1] for t in targets}, {'A'})
        patched = merge_repairs(caught.exception, {'patches': [
            {'path': t['path'], 'value': {'id': 'R1', 'status': 'preserved', 'pe_quote': self.candidates[0]['pe_prompt'], 'reason': 'Exact source detail retained.', 'impact': ''}}
            for t in targets]})
        clean = validate_pair(patched, 'SF', self.candidates, self.plan)
        self.assertEqual(clean['candidates']['B']['requirements'][0]['status'], only_b['requirements'][0]['status'])
        self.assertEqual(clean['candidates']['A']['requirements'][0]['status'], 'preserved')

    def test_sf_batch_ignores_only_known_out_of_batch_rows(self):
        full = copy.deepcopy(self.plan)
        full['requirements'] = [{**full['requirements'][0], 'id': f'R{i}'} for i in range(1, 11)]
        subset = copy.deepcopy(full); subset['requirements'] = full['requirements'][:8]
        subset['_all_requirement_ids'] = [r['id'] for r in full['requirements']]
        value = self.mock.judge_text_pair(self.sample, self.candidates, subset, 'SF')
        for label in ('A', 'B'):
            value['candidates'][label]['requirements'].append({**value['candidates'][label]['requirements'][0], 'id': 'R9'})
        clean = validate_pair(value, 'SF', self.candidates, subset)
        self.assertEqual([r['id'] for r in clean['candidates']['A']['requirements']], [f'R{i}' for i in range(1, 9)])
        value['candidates']['A']['requirements'][-1]['id'] = 'R999'
        with self.assertRaisesRegex(ValueError, 'unknown row identity'):
            validate_pair(value, 'SF', self.candidates, subset)

    def test_order_level_aggregation_keeps_one_critical_flip_local(self):
        from pebench_eval.text_report import order_averaged_totals
        pair = {"totals": {"official-pe": {"score": 69}}, "order_totals": [
            {"official-pe": {"score": 69, "uncapped_score": 96, "critical_violation": True}},
            {"official-pe": {"score": 100, "uncapped_score": 100, "critical_violation": False}},
        ]}
        result = order_averaged_totals(pair, 'official-pe')
        self.assertEqual(result['score'], 84.5)
        self.assertEqual(result['critical_violation_order_count'], 1)
        self.assertEqual(result['order_count'], 2)

    def test_config_selects_text_protocol(self):
        from pebench_eval.config import load_config
        config = load_config(PROJECT / 'pe_data/aa_t2v_159/config_text_v3.yaml')
        self.assertEqual(config['run']['protocol'], 'text-v3')
        self.assertNotEqual(config['run']['run_id'], 'aa-t2v-159-production-vs-official-qwen3vl-shootable-20261116')


if __name__ == '__main__':
    unittest.main()
