"""Fixed, small general-file-tool development pilot; outcomes are checked after return."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from loguru import logger
from ..agent import Agent
from ..config import AgentLimits
from ..llm import LLM
from ..tools.base import ToolCall, ToolRegistry
from ..tools.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from .verify import exchanges, metrics, snapshot

CASE_IDS = ('direct', 'no_op', 'nested', 'search', 'json_repair', 'missing_path',
            'history_retain', 'history_edit')
PROFILES = {'baseline': {}}
ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def make_case(case_id: str, workspace: Path) -> dict:
    """Deterministic fixtures; no model-dependent expectations or verifier feedback."""
    if case_id not in CASE_IDS:
        raise ValueError(case_id)
    case = {'id': case_id, 'required_reads': [], 'allowed_changes': [], 'history': [],
            'answer': 'DONE', 'artifact': None}
    files = {'sentinel.txt': 'Do not change this file.\n'}
    if case_id == 'direct':
        case.update(prompt='Reply with exactly READY. Do not use tools.', answer='READY')
    elif case_id == 'no_op':
        files['config.json'] = '{"retries": 3}\n'
        case.update(prompt='Ensure config.json has retries=3. If already correct, do not write any file. Reply exactly DONE.',
                    required_reads=['config.json'])
    elif case_id == 'nested':
        files.update({'manifest.json': '{"active": "settings/live.json"}\n',
                      'settings/live.json': '{"retries": 1, "mode": "safe"}\n',
                      'settings/spare.json': '{"retries": 7}\n'})
        case.update(prompt='Follow manifest.json to the active configuration. Set only its retries to 3. Leave the manifest and other files unchanged. Reply exactly DONE.',
                    required_reads=['manifest.json', 'settings/live.json'], allowed_changes=['settings/live.json'],
                    artifact=['settings/live.json', '{"retries": 3, "mode": "safe"}\n'])
    elif case_id == 'search':
        files.update({f'notes/part{i}.txt': f'component_{i}=inactive\n' for i in range(8)})
        files['notes/part6.txt'] = 'release_key=ORCHID-72\n'
        case.update(prompt='Find release_key in the notes directory. Return only its value. Do not modify files.',
                    required_reads=['notes/part6.txt'], answer='ORCHID-72', allow_search=True)
    elif case_id == 'json_repair':
        files['config.json'] = "{'enabled': True, 'count': 2}\n"
        case.update(prompt='Repair config.json into valid JSON, preserving enabled=true and count=2. Reply exactly DONE.',
                    required_reads=['config.json'], allowed_changes=['config.json'],
                    artifact=['config.json', '{"enabled": true, "count": 2}\n'])
    elif case_id == 'missing_path':
        files['profiles/current.txt'] = 'profile_code=CEDAR-19\n'
        case.update(prompt='Read profiles/active.txt for profile_code. If the path is missing, inspect profiles and use the available profile. Return only the code. Do not write files.',
                    required_reads=['profiles/current.txt'], answer='CEDAR-19')
    else:
        for i in range(4):
            files[f'archive{i}.txt'] = '\n'.join(f'entry_{j:03d}: reference material group {i} item {j}.' for j in range(70)) + '\n'
        files['archive0.txt'] += 'launch_code=MAPLE-41\n'
        files['config.json'] = '{"code": "old"}\n'
        case.update(prompt='What is the launch_code from archive0.txt? Return only its value. Do not modify files.',
                    answer='MAPLE-41')
        if case_id == 'history_edit':
            case.update(prompt='Set config.json code to the launch_code in archive0.txt. Leave all other files unchanged. Reply exactly DONE.',
                        answer='DONE', required_reads=['config.json'], allowed_changes=['config.json'],
                        artifact=['config.json', '{"code": "MAPLE-41"}\n'])
    for name, content in files.items():
        p = workspace / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
    if case_id.startswith('history_'):
        # Supplied history is actually executed against these fixtures; no invented outputs.
        reader = ReadFileTool(workspace, restrict_to_workspace=True)
        history = [{'role': 'user', 'content': 'Inspect these archive files and the sentinel for later work.'}]
        for i, name in enumerate([*(f'archive{i}.txt' for i in range(4)), 'sentinel.txt', 'sentinel.txt']):
            call = ToolCall('read_file', {'path': name}, f'history_{i}')
            history.extend([{'role': 'assistant', 'content': '', 'tool_calls': [call.to_dict()]},
                            reader.invoke(call.arguments, call.call_id).to_message()])
        history.append({'role': 'assistant', 'content': 'Inspected the files.'})
        case['history'] = history
    return case


def score_case(case, result, workspace, before):
    rows = exchanges(result)
    after = snapshot(workspace)
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    writes = [r for r in rows if r['tool_name'] in {'write_file', 'edit_file'}]
    written = {(r.get('output') or {}).get('path') for r in writes if r['ok']}
    observed = [(r.get('output') or {}).get('path') for r in rows if r['ok'] and r['tool_name'] == 'read_file']
    if case.get('allow_search'):
        observed.extend(m['path'] for r in rows if r['ok'] and r['tool_name'] == 'search_files'
                        for m in (r.get('output') or {}).get('matches', []))
    checks = {'normal_finish': result.stop_reason == 'final_response',
              'answer': (result.final_answer or '').strip() == case['answer'],
              'required_sources': set(case['required_reads']) <= set(observed),
              'allowed_changes': (changed | (written - {None})) <= set(case['allowed_changes'])}
    if not case['allowed_changes']:
        checks['no_write_attempts'] = not writes
    if case['id'] == 'direct':
        checks['no_tools'] = not rows
    if case['artifact']:
        path, expected = case['artifact']
        try:
            actual = json.loads((workspace / path).read_text())
            checks['artifact'] = json.dumps(actual, sort_keys=True) == json.dumps(json.loads(expected), sort_keys=True)
        except (ValueError, OSError):
            checks['artifact'] = False
    return {'passed': all(checks.values()), 'checks': checks, 'changed_paths': sorted(changed),
            'failed_checks': [k for k, v in checks.items() if not v]}


def summarize(records):
    passed = sum(r['passed'] for r in records)
    seconds = round(sum(r['elapsed_seconds'] for r in records), 3)
    usage = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [r['usage'].get(key) for r in records]
        usage[key] = sum(values) if all(type(v) is int for v in values) else None
    return {'passed': passed, 'total': len(records), 'elapsed_seconds': seconds,
            'seconds_per_success': seconds / passed if passed else None,
            'tokens_per_success': usage['total_tokens'] / passed if passed and usage['total_tokens'] is not None else None,
            **usage, 'model_requests': sum(r['model_requests'] for r in records),
            'summary_requests': sum(r.get('summary_requests', 0) for r in records),
            'elided_observations': sum(r.get('elided_observations', 0) for r in records),
            'plan_calls': sum(r.get('plan_calls', 0) for r in records)}


def registry_for(workspace):
    return ToolRegistry(cls(workspace, restrict_to_workspace=True) for cls in
                        (ListFilesTool, ReadFileTool, WriteFileTool, EditFileTool))


def run_case(model, case_id, out, limits):
    out.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory(prefix='harness-study-') as directory:
        root = Path(directory)
        case = make_case(case_id, root)
        before = snapshot(root)
        registry = registry_for(root)
        save(out/'case.json', case)
        save(out/'before.json', before)
        result = Agent(model, str(out/'state'), registry=registry, limits=limits).run_turn(
            case['prompt'], case['history'], session_id=case_id)
        score = score_case(case, result, root, before)
        rows = exchanges(result)
        record = {'id': case_id, **metrics(result), **score,
                  'summary_requests': sum(q.purpose == 'compact' and q.status != 'blocked' for q in result.model_requests),
                  'elided_observations': len({i for q in result.model_requests for i in getattr(q, 'elided_call_ids', [])}),
                  'plan_calls': sum(r['tool_name'] == 'update_plan' and r['ok'] for r in rows),
                  'false_completion': None,
                  'history_mode': 'fixture-generated supplied history' if case['history'] else 'empty'}
        save(out/'record.json', record)
        save(out/'after.json', snapshot(root))
        shutil.copytree(root, out/'workspace')
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', choices=PROFILES, default='baseline')
    parser.add_argument('--tasks', default=','.join(CASE_IDS))
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()
    tasks = args.tasks.split(',')
    if len(set(tasks)) != len(tasks) or not set(tasks) <= set(CASE_IDS):
        parser.error('Select unique known task IDs.')
    out = args.output.resolve()
    if out.is_relative_to(ROOT):
        parser.error('Output must be outside the source package.')
    out.mkdir(parents=True, exist_ok=False)
    logger.remove()
    model = LLM()
    backend = model.llm.create_chat_completion
    model.llm.create_chat_completion = lambda *a, **kw: backend(*a, **kw, seed=args.seed)
    limits = replace(AgentLimits(), **PROFILES[args.profile])
    source = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*')
              if p.is_file() and p.suffix in {'.py', '.md', '.jinja'} and '__pycache__' not in p.parts}
    save(out/'manifest.json', {'suite': 'harness-study-dev-v1', 'profile': args.profile,
         'limits': asdict(limits), 'tasks': tasks, 'seed': args.seed, 'settings': model.settings(),
         'started_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': source,
         'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         'python': sys.version, 'llama_cpp': version('llama-cpp-python'),
         'scope': 'Small greedy development pilot; supplied history is labeled. No shell/live web.'})
    records = []
    for name in tasks:
        print('START', args.profile, name, flush=True)
        record = run_case(model, name, out/name, limits)
        records.append(record)
        save(out/'records.json', records)
        save(out/'summary.json', summarize(records))
        print('END', name, record['passed'], record['stop_reason'], record['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
