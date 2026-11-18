from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from .providers import MockProvider, OpenAICompatibleProvider, ProviderBundle
from .runner import select_cases

ROOT = Path(__file__).resolve().parents[1]


def _mapping(value: Any, name: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f'{name}: unknown options: {sorted(unknown)}')
    return value


def load_config(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg = _mapping(value, 'config', {'paths', 'selection', 'run', 'provider'})
    paths = _mapping(cfg.get('paths', {}), 'paths', {'data_root', 'output_root', 'candidate_file', 'template_file'})
    selection = _mapping(cfg.get('selection', {}), 'selection', {'modes', 'case_keys', 'task_categories', 'task_types', 'limit'})
    run = _mapping(cfg.get('run', {}), 'run', {'run_id', 'method', 'inspect_content', 'resume', 'max_workers', 'protocol', 'checkpoint_source'})
    provider = _mapping(cfg.get('provider', {}), 'provider', {'type', 'timeout_seconds', 'max_tokens', 'requests_per_minute', 'trust_env', 'judge', 'image_facts', 'video_facts'})
    for section, values in [('paths', paths), ('run', run)]:
        for key, item in values.items():
            if key == 'checkpoint_source' and isinstance(item, list):
                if not item or any(not isinstance(x, str) or not x.strip() for x in item):
                    raise ValueError('run.checkpoint_source must contain nonempty paths')
            elif key in {'inspect_content', 'resume'}:
                if not isinstance(item, bool): raise ValueError(f'{section}.{key} must be boolean')
            elif section == 'run' and key == 'max_workers':
                continue
            elif item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f'{section}.{key} must be a non-empty string or null')
    for key in ('modes', 'case_keys', 'task_categories', 'task_types'):
        items = selection.get(key, [])
        if not isinstance(items, list) or any(not isinstance(x, str) or not x.strip() for x in items):
            raise ValueError(f'selection.{key} must be a list of non-empty strings')
    invalid = set(selection.get('modes', [])) - {'t2v', 'i2v', 'r2v'}
    if invalid: raise ValueError(f'unsupported modes {sorted(invalid)}; available: t2v, i2v, r2v')
    limit = selection.get('limit')
    if limit is not None and (type(limit) is not int or limit < 1): raise ValueError('selection.limit must be a positive integer or null')
    if provider.get('type', 'mock') not in {'mock', 'real'}: raise ValueError('provider.type must be mock or real')
    if 'trust_env' in provider and not isinstance(provider['trust_env'], bool): raise ValueError('provider.trust_env must be boolean')
    if run.get('protocol', 'legacy') not in {'legacy', 'text-v3'}:
        raise ValueError('run.protocol must be legacy or text-v3')
    if 'max_workers' in run and (type(run['max_workers']) is not int or not 1 <= run['max_workers'] <= 128):
        raise ValueError('run.max_workers must be an integer in [1, 128]')
    for key, low, high in [('timeout_seconds', 1, 3600), ('max_tokens', 256, 65536), ('requests_per_minute', 0, 10000)]:
        if key in provider and (type(provider[key]) is not int or not low <= provider[key] <= high):
            raise ValueError(f'provider.{key} must be an integer in [{low}, {high}]')
    for section in ('judge', 'image_facts', 'video_facts'):
        if section not in provider or provider[section] is None: continue
        settings = _mapping(provider[section], f'provider.{section}', {'endpoint', 'model', 'api_key_env', 'api_key_file', 'public_base_url'})
        for key, item in settings.items():
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f'provider.{section}.{key} must be a non-empty string or null')
        if section != 'video_facts' and 'public_base_url' in settings: raise ValueError(f'provider.{section}.public_base_url is only for video_facts')
    for key, item in paths.items():
        if item and not Path(item).expanduser().is_absolute(): paths[key] = str((path.parent / item).resolve())
    return cfg


def filtered_cases(cases, selection: dict):
    modes = set(selection.get('modes') or [])
    keys = set(selection.get('case_keys') or [])
    categories = set(selection.get('task_categories') or [])
    types = set(selection.get('task_types') or [])
    missing = keys - {c.key for c in cases}
    if missing: raise ValueError(f'unknown case keys: {sorted(missing)}')
    selected = [c for c in cases if (not keys or c.key in keys) and
                (not categories or c.task_category in categories) and (not types or c.task_type in types)]
    selected = select_cases(selected, modes or None, selection.get('limit'))
    if not selected: raise ValueError('no cases selected')
    return selected


def provider_from_config(settings: dict):
    if settings.get('type', 'mock') == 'mock': return MockProvider()
    def client(values, name, video=False):
        if not isinstance(values, dict): raise ValueError(f'provider.{name} is required')
        endpoint, model = values.get('endpoint'), values.get('model')
        key = os.environ.get(values.get('api_key_env', '')) if values.get('api_key_env') else None
        if not key and values.get('api_key_file'):
            p = Path(values['api_key_file']).expanduser()
            if not p.is_absolute(): p = ROOT / p
            key = p.read_text(encoding='utf-8').strip()
        if not endpoint or not model or not key: raise ValueError(f'provider.{name} requires endpoint, model and a non-empty key from api_key_env or api_key_file')
        return OpenAICompatibleProvider(endpoint=endpoint, model=model, api_key=key,
            timeout_seconds=settings.get('timeout_seconds', 300), max_tokens=settings.get('max_tokens', 8192),
            requests_per_minute=settings.get('requests_per_minute', 30), trust_env=settings.get('trust_env', True), name=name,
            media_public_base_url=values.get('public_base_url') if video else None)
    judge = client(settings.get('judge'), 'judge')
    image = client(settings['image_facts'], 'image_facts') if settings.get('image_facts') else judge
    video = client(settings['video_facts'], 'video_facts', True) if settings.get('video_facts') else None
    return ProviderBundle(image_facts=image, video_facts=video, judge_model=judge)
