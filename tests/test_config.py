import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pebench_eval.config import filtered_cases, load_config, provider_from_config

class ConfigTests(unittest.TestCase):
    def test_selection_intersection_before_limit(self):
        cases = [SimpleNamespace(key='t2v:001', mode='t2v', task_category='Music', task_type='Singing'),
                 SimpleNamespace(key='t2v:002', mode='t2v', task_category='Speech', task_type='Dialogue'),
                 SimpleNamespace(key='i2v:001', mode='i2v', task_category='Speech', task_type='Dialogue')]
        picked = filtered_cases(cases, {'modes': ['t2v'], 'task_categories': ['Speech'], 'case_keys': ['t2v:002', 'i2v:001'], 'limit': 1})
        self.assertEqual([c.key for c in picked], ['t2v:002'])

    def test_config_rejects_invalid_mode_and_options(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1] / '.test-tmp') as d:
            p = Path(d) / 'bad.yaml'
            p.write_text('selection:\n  modes: [t2va]\n')
            with self.assertRaisesRegex(ValueError, 'unsupported modes'): load_config(p)
            p.write_text('selection:\n  limitt: 1\n')
            with self.assertRaisesRegex(ValueError, 'unknown options'): load_config(p)

    def test_default_provider_and_secret_do_not_enter_metadata(self):
        cfg = load_config(Path(__file__).resolve().parents[1] / 'config.yaml')
        with patch.dict('os.environ', {'PEBENCH_JUDGE_API_KEY': 'test-secret'}):
            p = provider_from_config(cfg['provider'])
        self.assertEqual(p.judge_model.model, 'gpt-5.4')
        self.assertIs(p.image_facts, p.judge_model)
        self.assertNotIn('test-secret', json.dumps(p.metadata()))
