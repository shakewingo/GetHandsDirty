"""Task-matched Stage 06 report; preserve old results and separate runtime revisions."""
from pathlib import Path
import json
import sys

from analyze import aggregate, activations, read


PROFILES = ('baseline', 'elision', 'planning', 'elision_planning')
TASKS = ['history_retain', 'history_edit']


def panel(root, profile):
    directory = root/profile
    manifest = read(directory/'manifest.json')
    rows = [r for r in read(directory/'records.json') if r['id'] in TASKS]
    assert [r['id'] for r in rows] == TASKS
    requests = []
    for row in rows:
        trace = read(next((directory/row['id']/'state/runs').glob('*.jsonl')))
        summaries = [q for q in trace['model_requests'] if q['purpose'] == 'compact']
        requests.append({'id': row['id'], 'requests': [
            {k:q.get(k) for k in ('purpose','status','budget','usage','error_message',
                                  'compact_before','compact_after','elided_call_ids')}
            for q in trace['model_requests']],
            'summaries_applied':sum(q.get('compact_after') is not None and not q.get('error_message')
                                    for q in summaries),
            'summaries_generated':sum(q['status'] != 'blocked' for q in summaries)})
    return manifest, {'totals':aggregate(rows), 'tasks':rows,
                      'activations':activations(directory, rows), 'requests':requests}


def compare(old_root, new_root):
    panels = {}
    reference = read(new_root/'baseline/manifest.json')
    for profile in PROFILES:
        old_manifest, old = panel(old_root, profile)
        manifest, new = panel(new_root, profile)
        assert manifest['tasks'] == TASKS
        for field in ('suite','settings','seed','source_sha256','git_revision'):
            assert manifest[field] == reference[field], (profile,field)
        for field in ('suite','settings','seed'):
            assert manifest[field] == old_manifest[field], (profile,field)
        assert manifest['search_enabled'] == old_manifest['search_enabled']
        for key, value in old_manifest['limits'].items():
            assert manifest['limits'][key] == value, (profile,key)
        for task in TASKS:
            for root in (old_root,new_root):
                assert read(root/profile/task/'before.json') == read(new_root/'baseline'/task/'before.json')
                assert read(root/profile/task/'case.json')['prompt'] == read(new_root/'baseline'/task/'case.json')['prompt']
        panels[profile] = {'old':old, 'merged':new, 'limits':manifest['limits'],
                          'new_limit_fields':{k:v for k,v in manifest['limits'].items()
                                              if k not in old_manifest['limits']}}
    return {'old_runtime':old_manifest['git_revision'], 'merged_runtime':reference['git_revision'],
            'settings':reference['settings'], 'seed':reference['seed'],
            'source_sha256':reference['source_sha256'], 'tasks':TASKS,
            'scope':'8 targeted development runs, single greedy sample per condition; old vs merged changes multiple upstream mechanisms, not an isolated reserve ablation.',
            'panels':panels}


if __name__ == '__main__':
    data = compare(Path(sys.argv[1]), Path(sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for name, p in data['panels'].items():
        print(name, 'old',p['old']['totals'],'merged',p['merged']['totals'])
