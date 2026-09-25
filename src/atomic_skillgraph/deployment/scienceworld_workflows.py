"""Authored partial capability fragments, NEVER official task-goal contracts.

Inventory labels identify reference assets, not benchmark task dispatch rules.
Every query/scope is supplied by the caller at the explicit identity entry.
"""
import copy
from ..core.bindings import BindingExpression
from ..core.contracts import CompositeOccurrence,CompositeSkill,TaskContract
from ..core.edges import GraphEdge
from ..core.refs import SkillRef
from .scienceworld_reference import reference_atomics


def reference_workflows():
    assets={i+1:row[0] for i,row in enumerate(reference_atomics())}
    # (arity, downstream steps). Search a/b/c has already produced concrete
    # entity+location outputs; no role reads another occurrence's Agent proposal.
    patterns=[
        (2,[(12,{'instrument':'b','target':'a'}),(20,{}),(6,{'entity':'a'})]),
        (2,[(29,{'source':'a','destination':'b'}),(20,{}),(6,{'entity':'b'})]),
        (2,[(16,{'thermometer':'b','target':'a'})]),
        (2,[(16,{'thermometer':'b','target':'a'}),(20,{}),(16,{'thermometer':'b','target':'a'})]),
        (3,[(10,{'left':'a','right':'b'}),(8,{'device':'c'})]),
        (3,[(10,{'left':'a','right':'b'}),(8,{'device':'c'}),(12,{'instrument':'c','target':'a'}),(9,{'device':'c'})]),
        (1,[(6,{'entity':'a'}),(7,{'entity':'a'})]),
        (2,[(12,{'instrument':'b','target':'a'}),(6,{'entity':'a'})]),
        (2,[(12,{'instrument':'b','target':'a'}),(20,{}),(6,{'entity':'a'})]),
        (1,[(5,{'entity':'a'}),(20,{}),(6,{'entity':'a'})]),
        (2,[(29,{'source':'a','destination':'b'}),(6,{'entity':'b'})]),
        (2,[(12,{'instrument':'b','target':'a'}),(20,{}),(6,{'entity':'a'})]),
        (1,[(6,{'entity':'a'}),(20,{}),(6,{'entity':'a'})]),
        (1,[(5,{'entity':'a'}),(20,{}),(6,{'entity':'a'})]),
        (1,[(20,{}),(6,{'entity':'a'}),(7,{'entity':'a'})]),
        (2,[(12,{'instrument':'b','target':'a'}),(6,{'entity':'a'})]),
        (3,[(3,{'entity':'a'}),(4,{'entity':'a','container':'b'}),(12,{'instrument':'c','target':'a'}),(6,{'entity':'a'})]),
        (1,[(6,{'entity':'a'}),(7,{'entity':'a'}),(20,{}),(6,{'entity':'a'})]),
    ]
    graphs=[]
    for number,(arity,steps) in enumerate(patterns,1):
        nodes=[CompositeOccurrence('entry','entry',assets[20+arity].ref,{})]
        edges=[]
        def linked(step,role,source,source_role):
            edges.append(GraphEdge(f'{source}_{source_role}_to_{step}_{role}','data_flow',source,step,source_role,role,'registered'))
            return BindingExpression('data_flow',source_step=source,source_role=source_role)
        for letter in 'abc'[:arity]:
            step='find_'+letter
            nodes.append(CompositeOccurrence(step,step,assets[18].ref,{
                'query':linked(step,'query','entry','query' if arity==1 else 'query_'+letter),
                'locations':linked(step,'locations','entry','locations')}))
        for index,(aid,mapping) in enumerate(steps):
            step=f'procedure_{index+1}'
            bindings={role:linked(step,role,'find_'+source,'entity') for role,source in mapping.items()}
            if aid==20:
                # A portable one-step observation interval. The optional entry
                # output is not falsely certified as a required producer.
                bindings['wait_steps']=BindingExpression('constant',constant=1)
            nodes.append(CompositeOccurrence(step,step,assets[aid].ref,bindings))
        names=[assets[18].summary]+[assets[a].summary for a,_ in steps]
        graphs.append(CompositeSkill(SkillRef(f'composite_scienceworld_reference_g{number:02d}','1.0.0'),
            'Partial capability workflow: '+' then '.join(names),nodes,[n.step_id for n in nodes],edges,[],
            TaskContract(copy.deepcopy(assets[steps[-1][0]].effects),validator_id='partial_capability_only'),
            {'steps':['Caller supplies semantic queries and authorized ordered rooms once at entry.',
                      'Execute explicit DataFlow; unavailable actions or missing witnesses fail closed.',
                      'This fragment does not choose an experiment, scientific answer or official goal.']},
            {},{'task_contract_covered':False},{'experiment_kind':'authored_reference','inventory_id':f'G{number:02d}',
                'completion_authority':{'kind':'partial_capability'},'task_contract_covered':False,'historical_execution_claimed':False} ))
    return graphs
