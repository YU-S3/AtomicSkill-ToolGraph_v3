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
           input_types=None, preconditions=(), notes=(), input_constraints=None):
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
    # A wildcard can prove an event happened, but cannot bind a fresh output.
    # Name each effect-witness output explicitly in the Atomic final contract;
    # intermediate ACTION expectations remain ordinary wildcard observations.
    effects = list(effects)
    for role, derivation in derivations.items():
        if derivation['kind'] != 'effect_witness':
            continue
        for index, effect in enumerate(effects):
            arg = derivation['argument_role']
            if effect.predicate == derivation['predicate'] and effect.args.get(arg) is None:
                effects[index] = SemanticPredicate(effect.predicate,
                    {**effect.args, arg: binding(role)}, effect.cardinality, effect.distinct_by, effect.effect_domain)
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
    if input_constraints:
        for role, constraints in input_constraints.items():
            tool.signature['properties'][role].update(constraints)
        tool.artifact['value_contract_version'] = 2
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
    return [inspected, room_search(), container_search()]


def _guard(selector, body, node_id):
    return {'op':'IF','node_id':node_id,'condition':{'op':'exists','match':selector},
            'then_branch':body}


def room_search():
    found = {'source':'semantic_evidence','where':{'predicate':'entity.discovered_at',
        'argument_role':'entity','semantic_compatible_with':{**value('query'),'semantic_type':'entity'},
        'location':value('scope',True)},'project':{'kind':'argument','role':'entity'},'distinct':True}
    sources = {'entity':value('found',True),'location':value('scope',True)}
    matches = {'op':'FOR_EACH','node_id':'joint_matches','collection_source':found,
        'iteration_variable':'found','max_iterations':1,
        'body':[{'op':'RETURN','node_id':'joint_result','output_sources':sources}]}
    reached = {'op':'FOR_EACH','node_id':'reached_scope','iteration_variable':'reached',
        'max_iterations':1,'collection_source':{'source':'semantic_evidence','where':{
            'predicate':'agent.at_location','argument_role':'location',
            'semantic_compatible_with':{**value('scope',True),'semantic_type':'entity'}},
            'project':{'kind':'argument','role':'location'},'distinct':True},
        'body':[_guard(found,[matches],'target_exists')]}
    program = [{'op':'FOR_EACH','node_id':'authorized_rooms','collection_source':value('locations'),
        'iteration_variable':'scope','max_iterations':16,'body':[
            {'op':'ACTION','node_id':'visit_scope','action_type':'TELEPORT',
             'argument_mapping':{'destination':argument('scope',True)},'expected_effects':[
                {'predicate':'agent.at_location','args':{'location':argument('scope',True)},'effect_domain':'world'}]},
            reached]}]
    result = author('discover_entity_in_authorized_rooms',['query','locations'],
        {'entity':('entity.discovered_at','entity',{}),'location':('entity.discovered_at','location',{})},
        [predicate('entity.discovered_at',entity='entity',location='location')],program,max_actions=16,
        input_types={'locations':'list'},input_constraints={'locations':{
            'items':{'type':'string'},'minItems':1,'maxItems':16,'uniqueItems':True}})
    atomic,_,tool = result
    tool.artifact['program'].pop()  # Exhausted scopes cannot invent a RETURN.
    atomic.inputs[0].required_resolution = 'semantic'
    atomic.validator_spec.update(output_semantic_constraints={'entity':{'compatible_with_input':'query'}},
        input_authorization={'locations':{'kind':'ordered_entity_scope','element_semantic_type':'entity',
            'min_items':1,'max_items':16,'unique_items':True}})
    return result


def container_search():
    found = {'source':'semantic_evidence','where':{'predicate':'entity.in_container',
        'argument_role':'entity','semantic_compatible_with':{**value('query'),'semantic_type':'entity'},
        'container':value('scope',True)},'project':{'kind':'argument','role':'entity'},'distinct':True}
    sources = {'entity':value('found',True),'container':value('scope',True),'location':value('location')}
    returned = {'op':'FOR_EACH','node_id':'contained_matches','collection_source':found,
        'iteration_variable':'found','max_iterations':1,
        'body':[{'op':'RETURN','node_id':'contained_result','output_sources':sources}]}
    def exact(kind):
        return {'source':'action_catalog','where':{'action_type':kind,'container':value('scope',True)},
                'project':{'kind':'argument','role':'container'},'distinct':True}
    def operate(kind,effect):
        return {'op':'ACTION','node_id':kind.lower(),'action_type':kind,
            'argument_mapping':{'container':argument('scope',True)},'expected_effects':[
                {'predicate':effect,'args':{'container':argument('scope',True),
                    **({'evidence':None} if effect in EVIDENCE else {})},
                 'effect_domain':'evidence' if effect in EVIDENCE else 'world'}]}
    program = [action('TELEPORT',{'destination':'location'},[predicate('agent.at_location',location='location')]),
        {'op':'FOR_EACH','node_id':'authorized_containers','collection_source':value('containers'),
         'iteration_variable':'scope','max_iterations':8,'body':[
             _guard(exact('OPEN'),[operate('OPEN','container.open')],'can_open'),
             _guard(exact('LOOK_IN'),[operate('LOOK_IN','container.inspected'),
                 _guard(found,[returned],'target_exists')],'can_inspect')]}]
    result = author('discover_entity_in_authorized_containers',['query','location','containers'],
        {'entity':('entity.in_container','entity',{}),'container':('entity.in_container','container',{}),
         'location':'location'},[predicate('entity.in_container',entity='entity',container='container')],
        program,max_actions=17,input_types={'containers':'list'},input_constraints={'containers':{
            'items':{'type':'string'},'minItems':1,'maxItems':8,'uniqueItems':True}})
    atomic,_,tool = result
    tool.artifact['program'].pop()
    atomic.inputs[0].required_resolution = 'semantic'
    atomic.validator_spec.update(output_semantic_constraints={'entity':{'compatible_with_input':'query'}},
        input_authorization={'containers':{'kind':'ordered_entity_scope','element_semantic_type':'entity',
            'min_items':1,'max_items':8,'unique_items':True}})
    return result


def authority(harness):
    return {'harness_profile':harness.profile_name, 'primitive_action_schema':harness.primitive_action_schema(),
        'semantic_predicate_schema':to_primitive(harness.semantic_predicate_schema()),
        'supported_grounding_constraints':['argument_exists','argument_concrete','harness_affordance'],
        'tool_ir_opcodes':['ACTION','IF','FOR_EACH','STOP_WHEN','RETURN'],
        'output_derivation_kinds':['input_identity','effect_witness'],
        **{kind+'_schema_version':3 for kind in ('atomic','implementation','tool','composite')}}


def workflow_entry(count):
    """Explicit caller intent transfer, never an entity-grounding shortcut."""
    import copy
    queries = ['query'] if count == 1 else [f'query_{c}' for c in 'abc'[:count]]
    optional = ['location','containers','wait_steps']
    roles = queries + ['locations'] + optional
    intent = 'bind_' + {1:'single',2:'dual',3:'triple'}[count] + '_query_workflow_entry'
    row = author(intent,roles,{r:r for r in roles},[],[],input_types={
        **{r:'entity' for r in queries},'locations':'list','containers':'list','wait_steps':'integer'},
        input_constraints={'locations':{'items':{'type':'string'},'minItems':1,'maxItems':16,'uniqueItems':True},
            'containers':{'items':{'type':'string'},'minItems':1,'maxItems':8,'uniqueItems':True},
            'wait_steps':{'minimum':0,'maximum':8}})
    atomic,impl,tool = row
    for spec in atomic.inputs + atomic.outputs:
        if spec.name in queries:
            spec.required_resolution = 'semantic'
        if spec.name in optional:
            spec.required = False
    for spec in (tool.signature,tool.interface['output_schema']):
        spec['required'] = [r for r in spec['required'] if r not in optional]
    # Optional values are never invented as defaults or returned as null.
    def branch(index, selected, suffix):
        if index == len(optional):
            return [{'op':'RETURN','node_id':'publish_entry_'+suffix,
                'output_sources':{r:value(r) for r in queries+['locations']+selected}}]
        role = optional[index]
        return [{'op':'IF','node_id':'has_'+role+'_'+suffix,
            'condition':{'source':'tool_input','field':role,'op':'exists'},
            'then_branch':branch(index+1,selected+[role],suffix+'t'),
            'else_branch':branch(index+1,selected,suffix+'f')}]
    tool.artifact['program'] = branch(0,[],'root')
    tool.artifact['evidence_outputs'] = []
    atomic.validator_spec['input_authorization'] = {
        'locations':{'kind':'ordered_entity_scope','element_semantic_type':'entity','min_items':1,'max_items':16,'unique_items':True},
        'containers':{'kind':'ordered_entity_scope','element_semantic_type':'entity','min_items':1,'max_items':8,'unique_items':True}}
    return row


def bounded_wait():
    effects = [predicate('time.progressed',evidence=None)]
    program = [{'op':'FOR_EACH','node_id':'wait_steps','iteration_variable':'step',
        'collection_source':{'source':'bounded_count','count':value('wait_steps')},'max_iterations':8,
        'body':[action('WAIT1',{},effects)]}]
    return author('wait_bounded_steps',['wait_steps'],{'evidence_ref':('time.progressed','evidence',{})},
        effects,program,max_actions=8,input_types={'wait_steps':'integer'},
        input_constraints={'wait_steps':{'minimum':0,'maximum':8}})


def compound_assets():
    """Compose public procedures; stop immediately when the environment is terminal."""
    import copy
    result = []
    for intent,base,action_type in (
        ('discover_and_acquire_in_rooms',room_search,'PICK_UP'),
        ('discover_and_acquire_in_containers',container_search,'PICK_UP'),
        ('discover_and_examine_in_rooms',room_search,'EXAMINE'),
        ('discover_and_read_in_rooms',room_search,'READ')):
        a,i,t = base()
        effect_name = {'PICK_UP':'agent.holds','EXAMINE':'entity.examined','READ':'entity.read'}[action_type]
        args = {'entity':argument('found',True),**({'evidence':None} if effect_name in EVIDENCE else {})}
        final = predicate(effect_name,entity='entity',**({'evidence':None} if effect_name in EVIDENCE else {}))
        a.effects.append(final)
        if effect_name in EVIDENCE:
            a.outputs.append(ParameterSpec('evidence_ref','evidence_ref'))
            a.validator_spec['output_derivations']['evidence_ref'] = {'kind':'effect_witness','predicate':effect_name,'argument_role':'evidence'}
            a.effects[-1] = SemanticPredicate(final.predicate,
                {**final.args, 'evidence': binding('evidence_ref')}, effect_domain=final.effect_domain)
        def append_action(nodes):
            rewritten = []
            for n in nodes:
                if n['op'] == 'RETURN':
                    rewritten.append({'op':'ACTION','node_id':'act_on_found','action_type':action_type,
                        'argument_mapping':{'entity':argument('found',True)},'expected_effects':[
                            {'predicate':effect_name,'args':args,'effect_domain':'evidence' if effect_name in EVIDENCE else 'world'}]})
                    if effect_name in EVIDENCE:
                        n['output_sources']['evidence_ref'] = {'source':'semantic_evidence','where':{
                            'predicate':effect_name,'entity':value('found',True)},'project':{'kind':'argument','role':'evidence'},'distinct':True}
                for branch_name in ('body','then_branch','else_branch'):
                    if branch_name in n: n[branch_name] = append_action(n[branch_name])
                rewritten.append(n)
            return rewritten
        t.artifact['program'] = append_action(t.artifact['program'])
        # Conservative bounds include the action even though RETURN ends search.
        from ..tooling.value_reference import worst_case_actions
        t.artifact['max_actions'] = worst_case_actions(t.artifact['program'])
        t.artifact['final_effects'] = to_primitive(a.effects)
        t.safety['allowed_action_types'].append(action_type)
        t.interface['output_schema'] = parameter_schema(a.outputs)
        a.ref = SkillRef('atomic_scienceworld_'+intent,'1.0.0')
        t.ref = ToolRef('tool_scienceworld_'+intent,'1.0.0')
        i.ref = SkillRef('impl_scienceworld_'+intent,'1.0.0')
        i.abstract_ref = a.ref
        i.tool_bindings = [ToolBinding(t.ref,'primary',{p.name:binding(p.name) for p in a.inputs},0)]
        i.execution_policy['output_mapping'] = {p.name:BindingExpression('tool_output',source_role=p.name,source_step='primary') for p in a.outputs}
        for asset in (a,i,t):
            asset.metadata = {**asset.metadata,'canonical_intent':intent}
        result.append((a,i,t))
    connect = [predicate('electrical.connected',left='left',right='right'),predicate('device.active',device='device')]
    result.append(author('connect_then_activate',['left','right','device'],{r:r for r in ('left','right','device')},connect,
        [action('CONNECT',{'left':'left','right':'right'},connect[:1]),action('ACTIVATE',{'device':'device'},connect[1:])],max_actions=2))
    pour = predicate('liquid.poured',source='source',destination='destination',evidence=None)
    mixed = predicate('container.mixed',container='destination',evidence=None)
    result.append(author('pour_then_mix',['source','destination'],{'evidence_ref':('container.mixed','evidence',{'container':'destination'})},[mixed],
        [action('POUR',{'source':'source','destination':'destination'},[pour]),action('MIX',{'container':'destination'},[mixed])],max_actions=2))
    measured = predicate('measurement.observed',subject='target',instrument='thermometer',evidence=None)
    wait_program = copy.deepcopy(bounded_wait()[2].artifact['program'][:-1])
    result.append(author('wait_then_measure_temperature',['wait_steps','thermometer','target'],
        {'evidence_ref':('measurement.observed','evidence',{'subject':'target','instrument':'thermometer'})},[measured],
        wait_program+[action('USE',{'instrument':'thermometer','target':'target'},[measured])],max_actions=9,
        input_types={'wait_steps':'integer'},input_constraints={'wait_steps':{'minimum':0,'maximum':8}}))
    return result


def reference_atomics():
    rows = basic_assets()+preparation_assets()+[bounded_wait()]+[workflow_entry(n) for n in (1,2,3)]+compound_assets()
    assert len(rows) == 30
    for index,row in enumerate(rows,1):
        for asset in row:
            asset.metadata['reference_inventory_id'] = f'A{index:02d}'
    return rows
