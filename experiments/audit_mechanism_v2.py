"""Offline conformance on original policy requests; bytes are not token claims."""
import argparse
import json
from pathlib import Path
from atomic_skillgraph.agents.decision_frame import project_decision_frame, expand_frame
from atomic_skillgraph.agents.runtime_policy_projection import digest
from atomic_skillgraph.core.serialization import atomic_write_json


def audit(trace_path, payload_dir):
    trace = json.loads(Path(trace_path).read_text(encoding='utf-8'))
    audits = [r for r in trace['metadata']['runtime_context_projection_audits'] if 'node_context' in r]
    occurrences = {r['occurrence_id'] for r in audits}
    found = []
    for path in sorted(Path(payload_dir).glob('*.json')):
        request = json.loads(path.read_text(encoding='utf-8'))['payload']
        contexts = [json.loads(m['content'].split('POLICY_CONTEXT_JSON\n')[-1])
            for m in request['messages'] if m['role'] == 'user' and 'POLICY_CONTEXT_JSON\n' in m['content']]
        for raw in contexts:
            if (raw.get('task_goal') != trace['task']['goal'] or
                    raw.get('execution_frame', {}).get('current_occurrence_id') not in occurrences):
                continue
            compact, record = project_decision_frame(raw, scope='node', native_tool_specs=request['tools'])
            frame = expand_frame(compact)
            before = raw['current_state_snapshot']
            preserved = (frame['goal']['task_qualifiers'] == raw['task_goal']
                and frame['goal']['node'] == before['current_atomic']
                and frame['goal']['semantic_constraints'] == raw['task_semantic_context']
                and frame['goal']['downstream'] == before['downstream_obligations']
                and frame['execution'].get('repeat') == raw['execution_frame'].get('repeat')
                and compact['current_action_catalog'] == raw['current_action_catalog'])
            found.append({'request_file': path.name, 'source_hash': digest(raw),
                'before_body_utf8_bytes': record['before_body_utf8_bytes'],
                'after_body_utf8_bytes': record['after_body_utf8_bytes'],
                'qualifiers_identity_repeat_catalog_preserved': preserved,
                'shrunk': record['after_body_utf8_bytes'] < record['before_body_utf8_bytes']})
    return {'trace_id': trace['trace_id'], 'samples': found, 'count': len(found),
        'passed': len(found) == len(audits) and all(r['shrunk'] and r['qualifiers_identity_repeat_catalog_preserved'] for r in found),
        'measurement': 'utf8_bytes_not_tokens', 'llm_requests': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', required=True)
    parser.add_argument('--payload-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.trace, args.payload_dir)
    atomic_write_json(Path(args.output), result)
    print(json.dumps(result, ensure_ascii=False))
    raise SystemExit(0 if result['passed'] else 1)
