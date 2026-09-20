"""Cheap, reversible-in-trace elision of old re-readable observations in actor views."""
from copy import deepcopy
from dataclasses import replace
import json

from .context import ContextState


BODY_FIELDS = {'read_file': ('content',), 'list_files': ('entries',), 'search_files': ('matches',)}


def elide_old_outputs(state: ContextState, llm, schemas: dict, limits, before: dict) -> dict:
    """Publish replacements only when measured tokens shrink; raw evidence is untouched.

    This deliberately does not elide shell/web output, failed calls or partial reads.
    Summary generation uses the original raw observations, not the elided stubs.
    """
    if not limits.elision_enabled or before.get('count_method') != 'exact':
        return before
    window, reserve, count = (before.get(k) for k in ('window_tokens', 'response_reserve', 'prompt_tokens'))
    if any(type(x) is not int for x in (window, reserve, count)):
        return before
    available = window - reserve - limits.context_margin_tokens
    if available <= 0 or count < available * limits.elision_soft_ratio:
        return before
    replacements = dict(state.elided)
    for message in state.raw[state.covered:state.compact_boundary()]:
        call_id = message.get('tool_call_id')
        if message['role'] != 'tool' or call_id in replacements:
            continue
        try:
            row = json.loads(message['content'])
            output = row.get('output')
            fields = BODY_FIELDS.get(row.get('tool_name'), ())
            if (not row.get('ok') or not isinstance(output, dict) or not fields
                    or output.get('truncated') or output.get('next_offset') is not None
                    or output.get('next_pages') is not None):
                continue
            if sum(len(json.dumps(output.get(key, ''), ensure_ascii=False)) for key in fields) < limits.elision_min_chars:
                continue
            row = deepcopy(row)
            for key in fields:
                if key in row['output']:
                    row['output'][key] = [] if isinstance(row['output'][key], list) else ''
            row['output'].update(elided=True, elision_note='Old observation body omitted. Reread the source if needed; file state may have changed.')
            replacements[call_id] = json.dumps(row, ensure_ascii=False)
        except (ValueError, TypeError, AttributeError):
            continue
    if replacements == state.elided:
        return before
    candidate = replace(state, elided=replacements)
    after = llm.measure_context(candidate.messages(), schemas)
    if (after.get('count_method') == 'exact' and type(after.get('prompt_tokens')) is int
            and after['prompt_tokens'] < count):
        state.elided = replacements
        return after
    return before
