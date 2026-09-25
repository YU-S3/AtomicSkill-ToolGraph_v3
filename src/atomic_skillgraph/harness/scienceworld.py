"""ScienceWorld adapter: exact public actions, official score, checked replay."""
from __future__ import annotations
import copy
import importlib.metadata
import re
from ..core.bindings import BindingExpression, BindingExprKind
from ..core.contracts import TaskContract, SemanticPredicate, ContractSource
from ..core.refs import content_hash
from ..core.results import ValidationResult, AtomicEffectResolution, PrimitiveToolStep
from ..validation.contract_matcher import ExactContractMatcher
from .protocol import HarnessActionResult, HarnessRuntimeCheckpoint
from . import scienceworld_actions as actions
from . import scienceworld_public as public

def compatible(*, role, concrete_value, semantic_anchor, semantic_type):
    if concrete_value == semantic_anchor:
        return True
    if not isinstance(concrete_value, str) or not isinstance(semantic_anchor, str):
        return False
    # Whole public token phrase; no answer inference or hidden taxonomy.
    return bool(re.search(r'(?<!\w)' + re.escape(semantic_anchor.casefold()) + r'(?!\w)', concrete_value.casefold()))

class ScienceWorldValidatorChannel:
    validation_strength = 'public_action_evidence_and_official_score'

    def __init__(self):
        self.revision = 0
        self.done = self.won = False
        self.score = 0
        self.facts = []

    def snapshot(self):
        # Program selectors read the latest observation per exact subject tuple.
        # Historical references remain in self.facts and the immutable action
        # trace, and can still be validated when a caller explicitly binds one.
        current = {}
        for item in self.facts:
            if item.get('source_kind') == 'official_score':
                continue  # Validator-only goal authority is not policy evidence.
            identity = (item['predicate'], content_hash({k:v for k,v in item['args'].items() if k != 'evidence'}))
            if identity not in current or item['observed_at_revision'] > current[identity]['observed_at_revision']:
                current[identity] = item
        return {'revision': self.revision, 'done': self.done, 'won': self.won,
                'facts': copy.deepcopy(list(current.values())), 'validation_strength': self.validation_strength}

    @staticmethod
    def parts(effect):
        return (effect.predicate, effect.args, effect.cardinality, effect.distinct_by) if isinstance(effect, SemanticPredicate) else (
            effect['predicate'], effect.get('args', {}), effect.get('cardinality', 1), effect.get('distinct_by', ''))

    @staticmethod
    def expression(raw):
        if isinstance(raw, dict) and 'kind' in raw:
            raw = BindingExpression.from_dict(raw)
        if isinstance(raw, BindingExpression):
            if raw.kind == BindingExprKind.CONSTANT:
                return 'constant', raw.constant
            if raw.kind == BindingExprKind.SKILL_INPUT:
                return 'role', raw.source_role
            raise ValueError('Unsupported effect expression')
        if isinstance(raw, str) and raw.startswith('$'):
            return 'role', raw[1:]
        return 'constant', raw

    def candidates(self, effect, bindings):
        predicate, args, count, distinct = self.parts(effect)
        found = []
        for fact in self.facts:
            if fact['predicate'] != predicate:
                continue
            assignment = {}
            for role, raw in args.items():
                kind, value = self.expression(raw)
                actual = fact['args'].get(role)
                expected = bindings.get(value) if kind == 'role' else value
                if actual is None or (expected is not None and actual != expected):
                    break
                if kind == 'role':
                    if value in assignment and assignment[value] != actual:
                        break
                    assignment[value] = actual
            else:
                found.append((assignment, fact))
        if count > 1:
            if len(found) < count or (distinct and len({f['args'].get(distinct) for _, f in found}) < count):
                return []
        return found

    def validate_atomic_effect(self, request):
        effects, bindings = request.get('effects', []), request.get('bindings', {})
        for effect in effects:
            for raw in self.parts(effect)[1].values():
                kind, value = self.expression(raw)
                if kind == 'role' and bindings.get(value) in (None, ''):
                    return ValidationResult.fail('atomic', 'atomic_effect_unbound_role', value)
        groups = [self.candidates(e, bindings) for e in effects]
        passed = bool(groups) and all(groups)
        return ValidationResult('atomic', passed, checks={'public_effects': passed},
            witness_refs=[f['witness_ref'] for g in groups for _, f in g],
            failure_codes=[] if passed else ['atomic_effect_violation'])

    def resolve_atomic_effect(self, request):
        if request.get('current_revision') != self.revision:
            return AtomicEffectResolution(False, failure_code='stale_atomic_effect_witness')
        known = request.get('known_bindings', {})
        preferred = request.get('preferred_bindings', {})
        anchors = request.get('semantic_anchors', {})
        authoritative = {(f['predicate'], content_hash(f['args'])) for f in request.get('authoritative_evidence_facts', [])}
        combined = [(dict(known), [])]
        for effect in request.get('effects', []):
            next_groups = []
            for assignment, fact in self.candidates(effect, known):
                if any(k in preferred and preferred[k] != v for k, v in assignment.items()):
                    continue
                if any(k in anchors and not compatible(role=k, concrete_value=v, semantic_anchor=anchors[k], semantic_type='entity')
                       for k, v in assignment.items()):
                    continue
                if any(k not in known and k not in anchors and k not in preferred
                       and (fact['predicate'], content_hash(fact['args'])) not in authoritative
                       for k in assignment):
                    continue
                for prior, witnesses in combined:
                    if all(k not in prior or prior[k] == v for k, v in assignment.items()):
                        next_groups.append(({**prior, **assignment}, [*witnesses, fact['witness_ref']]))
            combined = next_groups
        distinct = {content_hash(a): (a, refs) for a, refs in combined}
        if not request.get('effects') or len(distinct) != 1:
            return AtomicEffectResolution(False, failure_code='atomic_effect_violation' if not distinct else 'atomic_effect_witness_ambiguous')
        resolved, refs = next(iter(distinct.values()))
        output_roles = {s['name'] if isinstance(s, dict) else s.name for s in request.get('output_specs', [])}
        outputs = {k: v for k, v in resolved.items() if k in output_roles}
        for item in request.get('output_identity', []):
            if item['input_role'] in resolved:
                outputs[item['output_role']] = resolved[item['input_role']]
        return AtomicEffectResolution(True, resolved_bindings=resolved, output_candidates=outputs,
            witness_refs=sorted(set(refs)), checks={'assignment_unique': True, 'witness_revision_current': True})

    def validate_task_contract(self, contract):
        return ValidationResult('task_contract', self.won, checks={'official_score_100': self.won},
            failure_codes=[] if self.won else ['task_contract_mismatch'])

class ScienceWorldAdapter:
    profile_name = 'scienceworld_v1'
    public_discovery_version = public.VERSION
    action_submission = 'exact_tuple'

    def __init__(self, *, split='train', max_steps=100, simplification='easy', version='1.2.3', env_factory=None):
        if version != '1.2.3' or simplification != 'easy' or max_steps != 100:
            raise ValueError('Formal ScienceWorld resource contract is 1.2.3/easy/100')
        self.split, self.max_steps, self.simplification = split, max_steps, simplification
        self._env_factory, self._env = env_factory, None
        self._validator = ScienceWorldValidatorChannel()
        self._catalog, self._prefix, self._action_facts = [], [], []
        self._revision = 0
        self._discovery = None
        self._container_inspections = []

    def initialize(self):
        if self._env is None:
            if self._env_factory:
                self._env = self._env_factory()
            else:
                if importlib.metadata.version('scienceworld') != '1.2.3':
                    raise ValueError('ScienceWorld package version changed')
                from scienceworld import ScienceWorldEnv
                self._env = ScienceWorldEnv(envStepLimit=self.max_steps)
        return len(self._env.tasks)

    def _close_backend(self):
        if self._env is not None:
            self._env.close()
            self._env = None

    def reset(self, task):
        # 1.2.x object UUIDs are JVM-global. Reusing a JVM after another
        # variation changes public referents, so every episode uses a fresh JVM,
        # exactly like rollback replay. Never remap or guess changed aliases.
        self._close_backend()
        self.initialize()
        self._task = copy.deepcopy(task)
        if task.metadata.get('resource_identity') and not self._env_factory:
            from .scienceworld_resource import verify_resource
            verify_resource(task.metadata['resource_identity'])
        name, variation = task.context['task_name'], task.context['variation_idx']
        self._env.load(name, variation, self.simplification, generateGoldPath=False)
        if variation not in getattr(self._env, 'get_variations_' + task.context['source_split'])():
            raise ValueError('Variation does not belong to declared official split')
        observation, info = self._env.reset()
        if (info['taskName'], info['variationIdx'], info['simplificationStr']) != (name, variation, self.simplification):
            raise ValueError('Reset resource identity mismatch')
        if task.goal and task.goal != info['taskDesc']:
            raise ValueError('Task description differs from manifest')
        self._revision, self._prefix, self._action_facts = 0, [], []
        self._container_inspections = []
        self._validator = ScienceWorldValidatorChannel()
        return self._refresh(observation, info, done=info['score'] == 100 or info['score'] < 0,
                             reward=info.get('reward', 0), accepted=True)

    def _refresh(self, observation, info, *, done, reward, accepted):
        self._frame = {'task_description': info['taskDesc'], 'observation': observation,
                       'look': info['look'], 'inventory': info['inv']}
        self._catalog = actions.catalog(self._env.get_valid_action_object_combinations_with_templates(), self._revision)
        facts, self._discovery = public.frame_evidence(self._frame, self._catalog, self._revision, self._task.task_id)
        # Snapshot facts come from this frame plus noncontradicted accepted actions.
        self._validator.facts = facts + copy.deepcopy(self._action_facts)
        self._validator.revision, self._validator.done = self._revision, bool(done)
        self._validator.score, self._validator.won = info['score'], info['score'] == 100
        if self._validator.won:
            family = public.GOALS[int(self._task.task_type.split('-')[0])-1]
            goal = public.fact(f'scienceworld.{family}_goal_satisfied', {'task':self._task.goal}, self._revision)
            goal.update(effect_domain='world',source_kind='official_score')
            self._validator.facts.append(goal)
        self._info = dict(info)
        text = observation + '\n\nLook:\n' + info['look'] + '\nInventory:\n' + info['inv']
        return HarnessActionResult(accepted, text, bool(done), self._validator.won, self._revision,
            self._catalog, {'environment_moves': info['moves']}, float(info['score']), float(reward))

    def execute_action(self, action_id, revision):
        if revision != self._revision:
            raise ValueError('stale_action_revision')
        matches = [a for a in self._catalog if a.action_id == action_id]
        if len(matches) != 1 or self._validator.done:
            raise ValueError('unknown_action_or_terminal_environment')
        spec = matches[0]
        observation, reward, done, info = self._env.step(spec.raw_action)
        accepted = public.action_accepted(observation)
        self._revision += 1
        if accepted:
            new, evidence = public.action_evidence(spec, observation, self._revision, self._task.task_id)
            # Transient world state is not a forever-valid action memory.
            self._action_facts = public.retain_action_evidence(self._action_facts, spec)
            self._action_facts.extend(new)
        result = self._refresh(observation, info, done=done, reward=reward, accepted=accepted)
        if spec.action_type == 'LOOK_IN':
            new, inspection = public.container_evidence(spec, observation, self._catalog,
                self._discovery, self._revision, self._task.task_id)
            self._container_inspections.append(inspection)
            # A new inspection replaces older claims for this container. The
            # immutable prefix retains the previous public observation.
            self._action_facts = [f for f in self._action_facts if not (
                f.get('source_kind') == public.VERSION + '/container_listing'
                and f['args'].get('container') == spec.arguments['container'])]
            self._action_facts.extend(new)
            facts, _ = public.frame_evidence(self._frame,self._catalog,self._revision,self._task.task_id)
            self._validator.facts = facts + copy.deepcopy(self._action_facts) + [
                f for f in self._validator.facts if f.get('source_kind') == 'official_score']
        self._prefix.append({'raw_action': spec.raw_action, 'action_type': spec.action_type,
            'arguments': spec.arguments, 'accepted': accepted, 'score': info['score'],
            'done': bool(done), 'state_digest': self._state_digest(), 'public_frame': self._normalized_frame(),
            'semantic_actions': self._semantic_actions()})
        return result

    def _normalized_frame(self):
        return {k: public.normalize_listing(v) for k, v in self._frame.items()}

    def _state_digest(self):
        return content_hash({'frame': self._normalized_frame(), 'score': self._validator.score, 'done': self._validator.done,
            'catalog': self._semantic_actions()})

    def _semantic_actions(self):
        # IDs originate in the public valid catalog (not an all-object LUT).
        # Multiple referent spellings for the same current object are aliases,
        # not different physics affordances. Fresh JVMs keep UUID allocation fixed.
        return sorted({(a.action_type, a.metadata['template_id'], tuple(a.metadata['obj_ids']))
                       for a in self._catalog})

    def capture_runtime_checkpoint(self):
        return HarnessRuntimeCheckpoint(copy.deepcopy(self._task), tuple(copy.deepcopy(self._prefix)), self._revision,
                                        self._state_digest(), {'score': self._validator.score})

    def restore_runtime_checkpoint(self, checkpoint):
        self._close_backend()
        result = self.reset(checkpoint.task)
        for event in checkpoint.accepted_prefix:
            spec = actions.resolve(self._catalog, event['action_type'], event['arguments'], self._revision)
            if spec.raw_action != event['raw_action']:
                raise RuntimeError('scienceworld_replay_raw_action_mismatch')
            result = self.execute_action(spec.action_id, spec.revision)
            if self._prefix[-1] != event:
                import json
                differences = {k: {'expected': event.get(k), 'actual': self._prefix[-1].get(k)}
                               for k in set(event) | set(self._prefix[-1]) if event.get(k) != self._prefix[-1].get(k)}
                if 'semantic_actions' in differences:
                    before, after = set(event['semantic_actions']), set(self._prefix[-1]['semantic_actions'])
                    differences['semantic_actions'] = {'removed': sorted(before-after), 'added': sorted(after-before)}
                raise RuntimeError('scienceworld_replay_determinism_failure: ' + json.dumps(differences))
        if self._revision != checkpoint.revision or self._state_digest() != checkpoint.state_digest:
            raise RuntimeError('scienceworld_replay_checkpoint_mismatch')
        result.metadata.update(restore_replay_action_count=len(checkpoint.accepted_prefix),
                               restored_digest=self._state_digest())
        return result

    def action_catalog(self): return list(self._catalog)
    def policy_surface(self):
        from .policy_surface import exact_tuple_surface
        return exact_tuple_surface(self)

    def resolve_policy_action(self, arguments):
        if set(arguments) != {'action_type', 'arguments'}:
            raise ValueError('Expected only action_type and arguments')
        return actions.resolve(self._catalog, arguments['action_type'], arguments['arguments'], self._revision)
    def public_discovery_frame(self): return self._discovery
    def public_container_inspection_frame(self):
        return {'version':'scienceworld.container-discovery.v1','revision':self._revision,
                'containers':copy.deepcopy(self._container_inspections)}
    def public_runtime_relation_facts(self): return self._discovery.relation_facts() if self._discovery else []
    def public_catalog_relation_schema(self): return []
    def validator_channel(self): return self._validator
    def semantic_predicate_schema(self): return public.predicate_schema()
    def primitive_action_schema(self): return actions.primitive_schema()
    def contract_matcher(self): return ExactContractMatcher()
    semantic_value_compatible = staticmethod(compatible)

    def task_contract(self, task):
        family = public.GOALS[int(task.task_type.split('-')[0]) - 1]
        return TaskContract([SemanticPredicate(f'scienceworld.{family}_goal_satisfied', {'task': task.goal})],
            source=ContractSource.ADAPTER_DERIVED, confidence=1.0, validator_id='scienceworld_official_score')

    def compile_primitive(self, primitive, bindings):
        values = {}
        for role, raw in primitive.argument_mapping.items():
            kind, value = self._validator.expression(raw)
            values[role] = bindings.get(value) if kind == 'role' else value
        try:
            return actions.resolve(self._catalog, primitive.action_type, values, self._revision)
        except ValueError as exc:
            # Harness protocol: unavailable grounded primitives are KeyError;
            # ToolRunner records an intrinsic rejection instead of crashing.
            raise KeyError(str(exc)) from exc

    def execute_primitive(self, primitive, bindings):
        spec = self.compile_primitive(primitive, bindings)
        return self.execute_action(spec.action_id, spec.revision)

    def supports_constraint(self, kind, verifier_id=''):
        return not verifier_id and kind in {'argument_exists', 'argument_concrete', 'harness_affordance'}

    def _match_action_event(self, event):
        return actions.resolve(self._catalog, event['action_type'], event['arguments'], self._revision)

    def replay_tool(self, task, tool, case):
        if tool.artifact_kind == 'tool_ir_v1':
            return False  # Production ToolRunner owns this IR.
        self.reset(task)
        for event in case.get('prefix', []):
            spec = self._match_action_event(event)
            if not self.execute_action(spec.action_id, spec.revision).accepted:
                return False
        for step in tool.artifact.get('steps', []):
            if not self.execute_primitive(PrimitiveToolStep(**step), case.get('bindings', {})).accepted:
                return False
        return self._validator.validate_atomic_effect({'effects': case.get('effects', []),
            'bindings': case.get('bindings', {})}).passed
