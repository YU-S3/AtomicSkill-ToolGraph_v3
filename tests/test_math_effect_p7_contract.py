from copy import deepcopy
from pathlib import Path
import pytest

from atomic_skillgraph.empirical import math_effect_contract as c
from atomic_skillgraph.empirical.math_effect_optimizer import apply_edits, receipt, verify_receipt
from atomic_skillgraph.empirical.math_effect_source import partition
from atomic_skillgraph.empirical.math_effect_selection import select
from atomic_skillgraph.empirical.contracts import PublicTask, digest
from atomic_skillgraph.empirical.bank import Bank

TOKENIZER='/mnt/d/T3S_exp/resources/tokenizers/deepseek-v4-flash/tokenizer.json'


def proposal(scope='global'):
    return {'category':'claim_strength','scope':scope,'goal':'Compare implications under explicit premises',
        'applicability':{'required_public_indicators':[] if scope=='global' else ['quantifier_syntax'],'excluded_indicators':[]},
        'steps':['Align stated assumptions before comparing the conclusions.','Check both directions separately; a stronger-looking statement may be false.'],
        'stop_condition':'Stop when a necessary premise cannot be established.','source_case_ids':['s1','s2'],
        'edit_rationale':'A testable cross-task hypothesis, not a proven explanation of the errors.'}


def candidate():return c.candidate(proposal(),['s1','s2'],{'response_hash':'real-response-hash'})


@pytest.mark.parametrize('change',[
    lambda v:v.update(steps=[None]),lambda v:v.update(category={}),lambda v:v.update(task_id='x'),
    lambda v:v.update(source_case_ids=[{},'s2']),lambda v:v.update(source_case_ids=['s1','external']),
    lambda v:v.update(goal='Select option B always'),lambda v:v.update(scope='local'),
    lambda v:v.update(steps=['Always select the stronger meta-option.'])])
def test_bad_candidate_paths(change):
    value=proposal();change(value)
    with pytest.raises(c.ContractError) as e:c.validate_candidate(value,['s1','s2'])
    assert all('field_path' in r for r in e.value.diagnostics)


def test_abstention_and_lineage():
    c.require({'decision':'no_change'},c.BATCH)
    p=candidate();edit={'decision':'edit','edits':[{'parent_id':p['id'],'rationale':'Narrow a failure condition.',
        'operations':[{'op':'replace','field':'stop_condition','value':'Stop if a necessary premise remains undecided.'}]}]}
    result=apply_edits(edit,{p['id']:p},['s1','s2'],{'response_hash':'edit'},TOKENIZER)[0]
    assert result['parent_id']==p['id'] and result['candidate']['source_case_ids']==['s1','s2']
    assert p==candidate()
    edit['edits'][0]['operations'][0]['field']='source_case_ids'
    with pytest.raises(c.ContractError):apply_edits(edit,{p['id']:p},['s1','s2'],{},TOKENIZER)


def test_public_partition_and_router():
    tasks=[{'task_id':str(i),'physical_key':str(i),'goal':'Let x be real','inputs':{'choices':[{'label':'A','text':'x>0'},{'label':'B','text':'x<0'}]},'split':'train'} for i in range(60)]
    meta={str(i):{'month':str(i%4),'theorem_type':['t'+str(i%3)]} for i in range(60)}
    a=partition(tasks,meta);b=partition(list(reversed(tasks)),meta)
    assert a==b and {k:len(v) for k,v in a['members'].items()}=={'proposal_train':28,'edit_feedback':12,'protected_gate':20}
    assert len(set(sum(a['members'].values(),[])))==60
    local=c.candidate(proposal('local'),['s1','s2'],{'response_hash':'r'})
    selected=select([candidate(),local],PublicTask(**tasks[0]),TOKENIZER)
    assert len(selected['selected'])==1 and selected['routing_http']==0


def trace(index,correct,skill=False):
    row={'task':{'physical_key':str(index)},'public_input_hash':'public'+str(index),'runtime_contract_hash':'contract',
        'score':{'hard':correct},'score_hash':digest(correct),'valid':True,'tokens':100,'http':1,
        'raw_usage':[{'total_tokens':100}],'request_ids':[str(index)+str(skill)],
        'response_hashes':['response'+str(index)+str(skill)],'selection':{'selected':[candidate()['id']] if skill else []}}
    row['trace_hash']=digest(row);return row


def good_receipt(n=12,name='edit_feedback'):
    return receipt([trace(i,False) for i in range(n)],[trace(i,i<2,True) for i in range(n)],
        [candidate()],digest(['s1','s2']),name,{'tasks':n,'min_net':2,'max_losses':1})


def test_pair_counters_hash_and_bank(tmp_path):
    feedback=good_receipt();gate=good_receipt(20,'protected_gate')
    assert feedback['wins']==2 and feedback['ties_wrong']==10 and verify_receipt(feedback)
    tampered=deepcopy(feedback);tampered['rows'][0]['on']['score']['hard']=False
    assert not verify_receipt(tampered)
    bank=Bank(tmp_path/'bank');bank.bind_choice_method(c.VERSION)
    with bank.db:bank.db.execute('INSERT INTO metadata VALUES(?,?)',('effect_source_keys','["s1", "s2"]'))
    bank.put('skill',c.asset(candidate()))
    with pytest.raises(ValueError):bank.put('skill',{'goal':'manual','guidance':'external'})
    published=c.asset(candidate(),[feedback,gate]);assert c.checked_effect_asset(published)
    bank.put('skill',published);bank.freeze(tmp_path/'frozen');bank.close()
    frozen=Bank(tmp_path/'frozen',readonly=True)
    assert len(frozen.all('skill'))==1;frozen.close()


def test_zero_gain_never_qualified():
    r=receipt([trace(i,True) for i in range(12)],[trace(i,True,True) for i in range(12)],
        [candidate()],digest(['s1','s2']),'edit_feedback',{'tasks':12,'min_net':2,'max_losses':1})
    assert not r['qualified'] and verify_receipt(r)
    with pytest.raises(ValueError):receipt([trace(0,True)],[trace(1,True)],[], 's','edit_feedback',{})
