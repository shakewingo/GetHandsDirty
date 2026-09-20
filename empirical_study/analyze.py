"""Validate a completed frozen panel and export compact, reviewable paired evidence."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def aggregate(rows):
    n = sum(r['passed'] for r in rows)
    seconds = round(sum(r['elapsed_seconds'] for r in rows), 3)
    totals = {}
    for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [r['usage'].get(key) for r in rows]
        totals[key] = sum(values) if all(type(v) is int for v in values) else None
    return {'passed': n, 'total': len(rows), 'seconds': seconds,
            'seconds_per_success': seconds/n if n else None,
            'tokens_per_success': totals['total_tokens']/n if n and totals['total_tokens'] is not None else None,
            **totals, 'model_requests': sum(r['model_requests'] for r in rows),
            'tool_attempts': sum(r['tool_attempts'] for r in rows),
            'summary_requests': sum(r['summary_requests'] for r in rows),
            'parse_errors': sum(r['parse_errors'] for r in rows),
            'invalid_calls': sum(r['invalid_calls'] for r in rows),
            'stop_reasons': dict(Counter(r['stop_reason'] for r in rows))}


def activations(panel, rows):
    counts = Counter()
    tasks = {}
    for row in rows:
        directory = panel/row['id']
        case = read(directory/'case.json')
        files = list((directory/'state/runs').glob('*.jsonl'))
        assert len(files) == 1, (directory, files)
        trace = read(files[0])
        tally = Counter()
        elided = set()
        for request in trace['model_requests']:
            elided.update(request.get('elided_call_ids', []))
            if request['status'] == 'blocked':
                tally[request['purpose'] + '_blocked'] += 1
        tally['elided_observations'] = len(elided)
        for message in trace['messages'][1+len(case['history']):]:
            if message['role'] == 'user' and message.get('content','').startswith('[Runtime feedback] Repeated'):
                tally['repeat_reminders'] += 1
            if message['role'] != 'tool':
                continue
            observation = json.loads(message['content'])
            if observation['tool_name'] in {'update_plan', 'search_files'} and observation['ok']:
                tally[observation['tool_name']] += 1
            output = observation.get('output')
            if isinstance(output, dict) and 'diagnostics' in output:
                tally['diagnostics'] += 1
                tally['diagnostic_'+output['diagnostics']['status']] += 1
        counts.update(tally)
        tasks[row['id']] = dict(tally)
    return {'total': dict(counts), 'by_task': tasks}


def no_op_audit(panel, rows):
    """Post hoc endpoint facts; never overwrite the frozen strict score."""
    row = next((r for r in rows if r['id'] == 'no_op'), None)
    if row is None:
        return None
    directory = panel/'no_op'
    trace = read(next((directory/'state/runs').glob('*.jsonl')))
    case = read(directory/'case.json')
    observations = [json.loads(m['content']) for m in trace['messages'][1+len(case['history']):]
                    if m['role'] == 'tool']
    mutations = [o for o in observations if o['tool_name'] in {'write_file','edit_file'}]
    unchanged = read(directory/'before.json') == read(directory/'after.json')
    confirmed = all(o['ok'] and (o.get('output') or {}).get('changed') is False for o in mutations)
    remaining = all(v for k,v in row['checks'].items() if k not in {'allowed_changes','no_write_attempts'})
    return {'strict_passed':row['passed'], 'snapshots_unchanged':unchanged,
            'other_checks_passed':remaining, 'mutation_results':[
                {'tool':o['tool_name'],'ok':o['ok'],'error_code':o.get('error_code'),
                 'changed':(o.get('output') or {}).get('changed')} for o in mutations],
            'all_mutations_explicitly_unchanged':confirmed,
            'conservative_sensitivity_passed':remaining and unchanged and confirmed}


def analyze(root):
    baseline_manifest = read(root/'baseline/manifest.json')
    baseline = {r['id']: r for r in read(root/'baseline/records.json')}
    panels = {}
    for profile in ('baseline','elision','planning','elision_planning','search','repeat','diagnostics'):
        panel = root/profile
        manifest = read(panel/'manifest.json')
        for field in ('suite','settings','source_sha256','seed','git_revision'):
            assert manifest[field] == baseline_manifest[field], (profile, field)
        rows = read(panel/'records.json')
        assert [r['id'] for r in rows] == manifest['tasks'], f'Incomplete panel: {profile}'
        for r in rows:
            assert read(panel/r['id']/'before.json') == read(root/'baseline'/r['id']/'before.json'), (profile,r['id'],'fixture')
        paired = [baseline[r['id']] for r in rows]
        wins = [r['id'] for r in rows if r['passed'] and not baseline[r['id']]['passed']]
        losses = [r['id'] for r in rows if not r['passed'] and baseline[r['id']]['passed']]
        panels[profile] = {'candidate': aggregate(rows), 'matched_baseline': aggregate(paired),
                          'wins': wins, 'losses': losses, 'activations': activations(panel,rows),
                          'no_op_post_hoc_audit':no_op_audit(panel,rows),
                          'limits':manifest['limits'], 'search_enabled':manifest['search_enabled'],
                          'tasks':[{k:r[k] for k in ('id','passed','checks','changed_paths','final_answer','failed_checks','stop_reason','model_requests',
                                    'elapsed_seconds','usage','summary_requests','tool_attempts','fault_encountered')}
                                   for r in rows]}
    return {'runtime_revision':baseline_manifest['git_revision'], 'suite':baseline_manifest['suite'],
            'settings':baseline_manifest['settings'],'seed':baseline_manifest['seed'],
            'source_manifest_sha256':sha256(json.dumps(baseline_manifest['source_sha256'],sort_keys=True).encode()).hexdigest(),
            'source_sha256':baseline_manifest['source_sha256'],
            'scope':'Single greedy development run per condition; no significance or generalization claim.',
            'no_op_audit_policy':'Post hoc: unchanged snapshots plus every mutation result explicitly ok/changed=False; other frozen checks retained. Errors/skips are not explicit no-change confirmations. Original scores never replaced.',
            'panels':panels}


if __name__ == '__main__':
    data = analyze(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for profile,panel in data['panels'].items():
        print(profile,json.dumps(panel['candidate']), 'wins',panel['wins'],'losses',panel['losses'],
              'activated',panel['activations']['total'])
