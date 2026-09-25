"""Authored ScienceWorld reference assets, not learned success evidence.

Programs are assembled using the production public IR. Concrete entities and
scientific answers are always caller inputs or public evidence, never constants.
"""
from ..core.bindings import BindingExpression, ToolBinding
from ..core.contracts import AbstractAtomicSkill, ParameterSpec, SemanticPredicate, ToolAsset, ImplementationAtom
from ..core.refs import SkillRef, ToolRef
from ..core.serialization import to_primitive
from ..tooling.entry_contract import parameter_schema
from ..tooling.ir import walk_program_nodes
from ..harness.scienceworld_public import WORLD, EVIDENCE


def binding(role):
    return BindingExpression('skill_input', source_role=role)


def value(role, local=False):
    return {'source': 'local_variable' if local else 'tool_input', 'field': role}


def argument(role, local=False):
    return {'kind': 'local_variable' if local else 'skill_input', 'source_role': role}


def predicate(name, **roles):
    return SemanticPredicate(name, {key: binding(role) if role is not None else None for key, role in roles.items()},
        effect_domain='world' if name in WORLD else 'evidence')


def evidence_source(name, role='evidence', **bindings):
    return {'source': 'semantic_evidence', 'where': {'predicate': name, **{r: value(v) for r,v in bindings.items()}},
        'project': {'kind': 'argument', 'role': role}, 'distinct': True}


def action(kind, arguments, effects, node=None):
    return {'op': 'ACTION', 'node_id': node or kind.lower(), 'action_type': kind,
        'argument_mapping': {r: argument(v) for r,v in arguments.items()},
        'expected_effects': to_primitive(effects)}


def author(intent, input_roles, output_roles, effects, program, *, output_sources=None, max_actions=1,
           input_types=None, preconditions=(), notes=()):
    types = input_types or {}
    inputs = [ParameterSpec(r, types.get(r, 'entity'), runtime_resolvable=True,
        required_resolution='concrete' if types.get(r, 'entity') == 'entity' else 'semantic') for r in input_roles]
    outputs, derivations, sources = [], {}, {}
    for role, origin in output_roles.items():
        if isinstance(origin, str):
            source = next(p for p in inputs if p.name == origin)
            outputs.append(ParameterSpec(role, source.semantic_type, required_resolution=source.required_resolution))
            derivations[role] = {'kind': 'input_identity', 'input_role': origin}
            sources[role] = value(origin)
        else:
            pred, arg, restrictions = origin
            outputs.append(ParameterSpec(role, 'evidence_ref' if arg == 'evidence' else 'entity',
                required_resolution='semantic' if arg == 'evidence' else 'concrete'))
            derivations[role] = {'kind': 'effect_witness', 'predicate': pred, 'argument_role': arg}
            sources[role] = evidence_source(pred, arg, **restrictions)
    sources.update(output_sources or {})
    metadata = {'canonical_intent': intent, 'authoring_source': 'scienceworld_reference_v1',
        'historical_execution_claimed': False, 'experiment_kind': 'authored_reference',
        'tool_builder_rationale': 'Caller-controlled procedure using exact public actions and witnessed outputs.'}
    atomic = AbstractAtomicSkill(SkillRef('atomic_scienceworld_' + intent, '1.0.0'), intent.replace('_',' '), inputs,
        outputs, list(preconditions), effects,
        {'validator_id':'harness_atomic_effect', 'identity_strict':True,
         'output_identity':[{'input_role': d['input_role'], 'output_role': r} for r,d in derivations.items() if d['kind']=='input_identity'],
         'output_derivations':derivations}, [],
        {'steps':['Resolve all required concrete inputs from the current public frame.',
                  'Execute the bounded procedure; keep scientific interpretation and answer choice with the Agent.'],
         'notes':list(notes)}, metadata)
    body = [*program, {'op':'RETURN','node_id':'return_outputs','output_sources':sources}]
    tool = ToolAsset(ToolRef('tool_scienceworld_' + intent, '1.0.0'), atomic.summary,
        parameter_schema(inputs), {'output_schema':parameter_schema(outputs),
        'entry_contract':{'conditions':to_primitive(preconditions),'grounding_constraints':[]}},
        'tool_ir_v1', {'schema_version':1,'max_actions':max_actions,'program':body,
        'final_effects':to_primitive(effects),'path_expectations':[],
        'evidence_outputs':[{'role':r, **s} for r,s in sources.items()]}, [],
        {'allowed_action_types':sorted({n['action_type'] for n in walk_program_nodes(body) if n['op']=='ACTION'}),
         'reviewed':True,'zero_llm':True,'terminal_interruptible':True}, metadata, metadata)
    implementation = ImplementationAtom(SkillRef('impl_scienceworld_'+intent,'1.0.0'), atomic.ref,
        [ToolBinding(tool.ref,'primary',{p.name:binding(p.name) for p in inputs},0)], [],
        {'mode':'serial','output_mapping':{p.name:BindingExpression('tool_output',source_role=p.name,source_step='primary') for p in outputs}},
        {'harness_profiles':['scienceworld_v1']}, {}, metadata=metadata)
    return atomic, implementation, tool


def basic_assets():
    # Public grammar and mechanical effects, not task-type routing.
    rows = [
        ('relocate_to_location','TELEPORT', {'destination':'destination'}, 'agent.at_location', {'location':'destination'}, {'location':'destination'}),
        ('open_container','OPEN', {'container':'container'}, 'container.open', {'container':'container'}, {'container':'container'}),
        ('acquire_entity','PICK_UP', {'entity':'entity'}, 'agent.holds', {'entity':'entity'}, {'entity':'entity'}),
        ('place_entity_in_container','MOVE', {'entity':'entity','container':'container'}, 'entity.in_container', {'entity':'entity','container':'container'}, {'entity':'entity','container':'container'}),
        ('focus_on_entity','FOCUS', {'entity':'entity'}, 'entity.focused', {'entity':'entity'}, {'entity':'entity'}),
        ('examine_entity','EXAMINE', {'entity':'entity'}, 'entity.examined', {'entity':'entity'}, {}),
        ('read_entity','READ', {'entity':'entity'}, 'entity.read', {'entity':'entity'}, {}),
        ('activate_device','ACTIVATE', {'device':'device'}, 'device.active', {'device':'device'}, {'device':'device'}),
        ('deactivate_device','DEACTIVATE', {'device':'device'}, 'device.inactive', {'device':'device'}, {'device':'device'}),
        ('connect_components','CONNECT', {'left':'left','right':'right'}, 'electrical.connected', {'left':'left','right':'right'}, {'left':'left','right':'right'}),
        ('disconnect_components','DISCONNECT', {'entity':'entity'}, 'electrical.disconnected', {'entity':'entity'}, {'entity':'entity'}),
        ('use_item_on_target','USE', {'instrument':'instrument','target':'target'}, 'instrument.used_on', {'instrument':'instrument','target':'target'}, {}),
        ('pour_into_container','POUR', {'source':'source','destination':'destination'}, 'liquid.poured', {'source':'source','destination':'destination'}, {'destination':'destination'}),
        ('mix_container_contents','MIX', {'container':'container'}, 'container.mixed', {'container':'container'}, {}),
        ('wait_one_step','WAIT1', {}, 'time.progressed', {}, {}),
        ('measure_temperature','USE', {'instrument':'thermometer','target':'target'}, 'measurement.observed', {'subject':'target','instrument':'thermometer'}, {'target':'target'}),
        ('power_target_device','ACTIVATE', {'device':'device'}, 'device.active', {'device':'device'}, {'device':'device'}),
    ]
    result = []
    for intent, kind, args, pred, roles, outs in rows:
        if pred in EVIDENCE:
            outs = {**outs, 'measurement_ref' if pred=='measurement.observed' else 'evidence_ref':(pred,'evidence',roles)}
            roles = {**roles, 'evidence':None}
        effects = [predicate(pred, **roles)]
        result.append(author(intent, list(dict.fromkeys(args.values())), outs, effects, [action(kind,args,effects)]))
    return result


def preparation_assets():
    inspect_effects = [predicate('scope.inspected', location='location', evidence=None)]
    inspected = author('inspect_current_scope', ['location'],
        {'evidence_ref':('scope.inspected','evidence',{'location':'location'})}, inspect_effects, [], max_actions=1,
        preconditions=[predicate('agent.at_location', location='location')],
        notes=['Uses the already exposed current public frame; it does not call a private observation API.'])
    found = {'source':'semantic_evidence','where':{'predicate':'entity.discovered_at',
        'argument_role':'entity','semantic_compatible_with':{**value('query'),'semantic_type':'entity'},
        'location':value('scope',True)}, 'project':{'kind':'argument','role':'entity'}, 'distinct':True}
    location = {'source':'semantic_evidence','where':{'predicate':'entity.discovered_at',
        'entity':value('found',True),'location':value('scope',True)},
        'project':{'kind':'argument','role':'location'},'distinct':True}
    sources = {'entity':value('found',True),'location':location}
    effects = [predicate('entity.discovered_at',entity='entity',location='location')]
    body = [{'op':'ACTION','node_id':'visit_scope','action_type':'TELEPORT',
             'argument_mapping':{'destination':argument('scope',True)},'expected_effects':[
                 {'predicate':'agent.at_location','args':{'location':argument('scope',True)},'effect_domain':'world'}]},
        {'op':'FOR_EACH','node_id':'joint_matches','collection_source':found,'iteration_variable':'found',
         'max_iterations':1,'body':[{'op':'RETURN','node_id':'joint_result','output_sources':sources}]}]
    program = [{'op':'FOR_EACH','node_id':'authorized_rooms','collection_source':value('locations'),
                'iteration_variable':'scope','max_iterations':8,'body':body}]
    discovery = author('discover_entity_in_authorized_scopes', ['query','locations','allow_open'],
        {'entity':('entity.discovered_at','entity',{}),'location':('entity.discovered_at','location',{})}, effects,
        program, output_sources=sources, max_actions=8, input_types={'locations':'list','allow_open':'bool'},
        notes=['Visit only caller-supplied locations. A room with unparsed descriptions is not an empty scope.',
               'Do not acquire or select an answer. Return only a jointly witnessed entity and location.'])
    atomic, _, tool = discovery
    # Exhausting authorized scopes has no successful RETURN. Loop locals must
    # not escape as invented outputs when no joint witness was found.
    tool.artifact['program'].pop()
    atomic.inputs[0].required_resolution = 'semantic'
    atomic.validator_spec.update(output_semantic_constraints={'entity':{'compatible_with_input':'query'}},
        input_authorization={'allow_open':{'kind':'caller_boolean'},'locations':{
        'kind':'ordered_entity_scope','element_semantic_type':'entity','min_items':1,'max_items':8,'unique_items':True}})
    tool.signature['properties']['locations'].update(items={'type':'string'},minItems=1,maxItems=8,uniqueItems=True)
    return [inspected, discovery]


def authority(harness):
    return {'harness_profile':harness.profile_name, 'primitive_action_schema':harness.primitive_action_schema(),
        'semantic_predicate_schema':to_primitive(harness.semantic_predicate_schema()),
        'supported_grounding_constraints':['argument_exists','argument_concrete','harness_affordance'],
        'tool_ir_opcodes':['ACTION','IF','FOR_EACH','STOP_WHEN','RETURN'],
        'output_derivation_kinds':['input_identity','effect_witness'],
        **{kind+'_schema_version':3 for kind in ('atomic','implementation','tool','composite')}}
