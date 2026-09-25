"""One public frame/parser for all five baseline methods."""
import json
from atomic_skillgraph.harness.scienceworld_actions import compact, resolve

class PolicyRejection(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(json.dumps({'protocol_errors': errors}, sort_keys=True))

class ScienceWorldTextPolicyProtocol:
    instruction = (
        'Interact with the public ScienceWorld environment. Return exactly one JSON object containing '
        'action_type and arguments, using one exact tuple from valid_actions_compact. '
        'No other fields, raw command, action index, markdown, or commentary are permitted. '
        'A unary catalog lists every allowed argument value; multi-argument catalogs list role-ordered tuples. '
        'WAIT and WAIT1 use an empty arguments object. Use only current public evidence; '
        'do not infer scientific answers from names or use hidden state. Complete the stated task within the action budget.')

    @staticmethod
    def retire_action_catalogs(messages):
        """Catalog tuples are revision-local, not historical task evidence.

        Preserve every observation and submitted action. Only the latest user
        frame supplied next retains its full executable action catalog.
        """
        for message in messages:
            if message.get('role') != 'user':
                continue
            try:
                frame = json.loads(message.get('content', ''))
            except (TypeError, ValueError):
                continue
            if isinstance(frame, dict) and 'valid_actions_compact' in frame:
                frame = dict(frame)
                del frame['valid_actions_compact']
                frame['historical_catalog_retired'] = True
                message['content'] = json.dumps(frame, ensure_ascii=False)

    @staticmethod
    def frame(harness):
        f = harness._frame
        return {'task': f['task_description'], 'observation': f['observation'],
            'look': f['look'], 'inventory': f['inventory'],
            'valid_actions_compact': compact(harness.action_catalog())}

    @staticmethod
    def parse(text, *, harness, revision):
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            raise PolicyRejection(['response must be one JSON object']) from None
        if not isinstance(data, dict):
            raise PolicyRejection(['response must be one JSON object'])
        errors = []
        required = {'action_type', 'arguments'}
        errors.extend(f'unexpected field: {k}' for k in sorted(set(data) - required))
        errors.extend(f'missing field: {k}' for k in sorted(required - set(data)))
        if 'action_type' in data and not isinstance(data['action_type'], str):
            errors.append('action_type must be a string')
        if 'arguments' in data and (not isinstance(data['arguments'], dict) or
                any(not isinstance(v, str) for v in data['arguments'].values())):
            errors.append('arguments must map roles to exact catalog strings')
        if revision != harness.validator_channel().revision:
            errors.append('stale_revision')
        if errors:
            raise PolicyRejection(errors)
        try:
            return resolve(harness.action_catalog(), data['action_type'], data['arguments'], revision)
        except ValueError as exc:
            raise PolicyRejection([str(exc)]) from exc
