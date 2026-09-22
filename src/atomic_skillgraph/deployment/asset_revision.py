"""Typed revision of the explicitly enumerated authored bank assets only."""
from __future__ import annotations

import copy
from dataclasses import replace

from ..agents.skill_guidance import normalize_guideline
from ..core.bindings import BindingExpression
from ..core.contracts import ParameterSpec, SemanticPredicate, TaskContract
from ..core.edges import GraphEdge
from ..core.refs import SkillRef, ToolRef
from ..core.serialization import to_primitive
from ..tooling.ir import walk_program_nodes

STEPS = {
    'C01': ['Use the caller-supplied location.', 'Navigate when required, open only when authorized, then inspect the location.', 'Return the declared location after the inspection effect is verified.'],
    'C02': ['Search only the supplied ordered location scope.', 'Use the current public affordances to identify a matching takeable object and its source.', 'Return the matched object and source without taking it.'],
    'C03': ['Use the supplied concrete object and source.', 'Navigate and perform authorized opening when needed, then take that same object.', 'Return the verified holding result without selecting a different object.'],
    'C04': ['Search only the caller-authorized ordered scope.', 'Select a publicly takeable object matching the supplied target and take it.', 'Return the concrete held object and the unchanged opening authorization.'],
    'C05': ['Use the held object and the supplied destination.', 'Navigate and perform authorized opening when needed, then place the same object.', 'Return the declared placement result.'],
    'C06': ['Use the held object and supplied cleaning station.', 'Reach the station and perform the available cleaning operation.', 'Return the same object with the verified cleaning result.'],
    'C07': ['Use the held object and supplied heating station.', 'Reach the station and perform the available heating operation.', 'Return the same object with the verified heating result.'],
    'C08': ['Use the held object and supplied cooling station.', 'Reach the station and perform the available cooling operation.', 'Return the same object with the verified cooling result.'],
    'C09': ['Use the held object, light and light location supplied by the caller.', 'Reach the location and perform the supported light operation when the goal is not already satisfied.', 'Return the verified observation result for that object and light.'],
}

def ref_map(assets):
    return {str(a.ref): replace(a.ref, version='1.0.1') for a in assets
            if a.metadata.get('authoring_source') == 'llm_bank_edit'
            and a.metadata.get('catalog_id') in {*STEPS, *(f'G{i:02}' for i in range(1, 7))}}

def _input(role):
    return BindingExpression('skill_input', source_role=role)

def _flow(role, step):
    return BindingExpression('data_flow', source_role=role, source_step=step)

def _fix_navigation(program):
    # Author the arrival guard using existing bounded semantic-evidence IR.
    # Absence of GO_TO is NOT arrival; the remainder runs only when a current
    # agent.at_location fact matches the exact caller-supplied destination.
    for index,node in enumerate(program):
        action = (node['then_branch'][0] if node['op']=='IF' and node['node_id'].endswith('go_if_available')
                  else node if node['op']=='ACTION' and node.get('action_type')=='GO_TO' else None)
        if action is not None:
            expression=action['argument_mapping']['destination']
            source='local_variable' if expression['kind']=='local_variable' else 'tool_input'
            anchor={'source':source,'field':expression['source_role'],'semantic_type':'entity'}
            navigate={'op':'IF','node_id':action['node_id']+'_when_available','condition':{
                'op':'exists','match':{'source':'action_catalog','where':{'action_type':'GO_TO',
                    'argument_role':'destination','semantic_compatible_with':anchor},
                    'project':{'kind':'argument','role':'destination'},'distinct':True}},
                'then_branch':[copy.deepcopy(action)],'else_branch':[]}
            tail=copy.deepcopy(program[index+1:])
            guard={'op':'FOR_EACH','node_id':action['node_id']+'_confirmed_arrival',
                'iteration_variable':'confirmed_arrival','max_iterations':1,
                'collection_source':{'source':'semantic_evidence','where':{'predicate':'agent.at_location',
                    'argument_role':'location','semantic_compatible_with':anchor},
                    'project':{'kind':'argument','role':'location'},'distinct':True},'body':tail}
            program[index:]=[navigate,guard]
            return
        for key in ('body','then_branch','else_branch'):
            if key in node:_fix_navigation(node[key])

def revise(kind, original, refs):
    """Return a detached 1.0.1 payload, never mutate any source object."""
    if str(original.ref) not in refs:
        return copy.deepcopy(original)
    asset = copy.deepcopy(original)
    asset.ref = refs[str(original.ref)]
    code = asset.metadata['catalog_id']
    asset.metadata['source_ref'] = str(original.ref)
    asset.metadata['release_revision'] = 'r103.bank-release.v1'
    if kind == 'atomic':
        asset.guideline = normalize_guideline({'steps': STEPS[code], 'notes': list(original.guideline.get('notes', []))[:2]},
            formal_roles=[p.name for p in [*asset.inputs, *asset.outputs]])
        for p in asset.inputs:
            p.runtime_resolvable = True
        if code == 'C02':
            asset.validator_spec['output_derivations'] = {
                role: {'kind':'effect_witness', 'predicate':'entity.discovered_at', 'argument_role': argument}
                for role, argument in [('object','entity'), ('source','location')]}
        if code == 'C04':
            asset.outputs = [p for p in asset.outputs if p.name != 'source']
            asset.validator_spec['output_derivations'] = {'object': {
                'kind':'effect_witness', 'predicate':'agent.holds', 'argument_role':'object'}}
        if code in {'C01','C02','C03','C04','C05'}:
            asset.outputs.append(ParameterSpec('allow_open','bool',required_resolution='semantic'))
            asset.validator_spec.setdefault('output_derivations', {})['allow_open'] = {'kind':'input_identity','input_role':'allow_open'}
            auth = {'allow_open':{'kind':'caller_boolean'}}
            if code in {'C02','C04'}:
                auth['locations'] = {'kind':'ordered_entity_scope','element_semantic_type':'entity',
                    'min_items':1,'max_items':8,'unique_items':True}
            asset.validator_spec['input_authorization'] = auth
    elif kind == 'tool':
        _fix_navigation(asset.artifact['program'])
        if code == 'C04':
            output = asset.interface['output_schema']
            output['properties'].pop('source', None)
            output['required'] = [r for r in output['required'] if r != 'source']
            asset.artifact['evidence_outputs'] = [e for e in asset.artifact['evidence_outputs'] if e['role'] != 'source']
            for node in walk_program_nodes(asset.artifact['program']):
                if node['op'] == 'RETURN': node['output_sources'].pop('source', None)
        if code in {'C01','C02','C03','C04','C05'}:
            output = asset.interface['output_schema']
            output['properties']['allow_open'] = {'type':'boolean'}
            output['required'].append('allow_open')
            asset.artifact['evidence_outputs'].append({'role':'allow_open','source':'tool_input','field':'allow_open'})
            for node in walk_program_nodes(asset.artifact['program']):
                if node['op'] == 'RETURN': node['output_sources']['allow_open'] = {'source':'tool_input','field':'allow_open'}
        if code in {'C02','C04'}:
            asset.signature['properties']['locations'].update(items={'type':'string'},minItems=1,maxItems=8,uniqueItems=True)
    elif kind == 'implementation':
        asset.abstract_ref = refs[str(asset.abstract_ref)]
        for binding in asset.tool_bindings:
            binding.tool_ref = refs[str(binding.tool_ref)]
        outputs = asset.execution_policy['output_mapping']
        if code == 'C04': outputs.pop('source', None)
        if code in {'C01','C02','C03','C04','C05'}:
            outputs['allow_open'] = BindingExpression('tool_output',source_role='allow_open',source_step='primary')
    elif kind == 'composite':
        occurrences=[]
        for occ in asset.occurrences:
            bindings=dict(occ.binding_specs)
            # Only known formal task-role correspondences are seeded.
            if 'target' in bindings: bindings['target'] = _input('object')
            if 'light' in bindings: bindings['light'] = _input('light_source')
            if code == 'G06' and occ.step_id.startswith('acquire'):
                bindings['object'] = _input('object')
                # Source instances are task-local unresolved roles, not task facts.
                bindings.pop('source', None)
            if occ.step_id.startswith('deliver'):
                producer = 'acquire1' if code == 'G06' else 'acquire'
                bindings['allow_open'] = _flow('allow_open',producer)
                asset.data_edges.append(GraphEdge(f'{code}_{occ.step_id}_allow_open','data_flow',producer,occ.step_id,'allow_open','allow_open','proposed'))
            if code == 'G06' and occ.step_id == 'acquire2':
                bindings['allow_open'] = _flow('allow_open','acquire1')
                asset.data_edges.append(GraphEdge(f'{code}_acquire2_allow_open','data_flow','acquire1','acquire2','allow_open','allow_open','proposed'))
            bindings={role:b for role,b in bindings.items()
                      if b.kind.value!='skill_input' or b.source_role in {'object','destination','light_source'}}
            occurrences.append(replace(occ,node_ref=refs[str(occ.node_ref)],binding_specs=bindings))
        asset.occurrences=occurrences
        asset.metadata['external_inputs'] = {
            o.step_id: {role: to_primitive(binding) for role, binding in o.binding_specs.items()
                        if binding.kind.value == 'skill_input'} for o in occurrences}
        # Publication makes these checked graph edges available through the
        # existing Active-graph resolver. Do not claim extractor execution or
        # invent evidence_refs for an authored edge.
        asset.data_edges=[replace(e,origin='existing_active',existing_edge_id=e.edge_id) for e in asset.data_edges]
        asset.dependency_edges=[replace(e,origin='existing_active',existing_edge_id=e.edge_id) for e in asset.dependency_edges]
        effects=[]
        if code in {'G02','G03','G04'}:
            effects.append(SemanticPredicate({'G02':'object.cleaned','G03':'object.heated','G04':'object.cooled'}[code],{'object':_input('object')}))
        if code == 'G05':effects.append(SemanticPredicate('object.observed_with',{'object':_input('object'),'light':_input('light_source')}))
        else:effects.append(SemanticPredicate('object.at_location',{'object':_input('object'),'location':_input('destination')},cardinality=2 if code=='G06' else 1,distinct_by='object' if code=='G06' else ''))
        asset.goal_contract=TaskContract(effects,
            cardinality_constraints=[{'predicate':'object.at_location','count':2,'distinct_by':'object','shared_roles':['location'],'composition_mode':'repeat_unit'}] if code=='G06' else [],
            identity_constraints=[{'left_role':'object','relation':'same_as','right_role':'object','scope':'task'}] if code in {'G02','G03','G04'} else [],
            validator_id='alfworld_v3_goal')
        asset.guideline=normalize_guideline({'steps':[
            'Acquire the authorized object and preserve its concrete identity.',
            'Execute each declared processing step before delivery or observation.',
            'Pass outputs only along the declared data-flow edges.',
            'For repeated work, acquire a distinct object for each iteration and preserve the shared destination.'
        ],'notes':['Missing scope, station or source inputs remain for the Agent to ground.']})
        asset.validator_spec['task_contract_covered']=False
        asset.metadata.pop('completion_authority',None)
    return asset
