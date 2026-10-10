"""The fixed LM1 review requires zero remote calls."""
from copy import deepcopy
import pytest
from test_lm1 import proposal,source,config,recompile,task,check
from test_cf2_contracts import http,response
from test_empirical import config_for
from atomic_skillgraph.empirical.choice_guidance import validate_proposal
from atomic_skillgraph.empirical.prompts import CHOICE_PROPOSAL
from atomic_skillgraph.empirical.system import EmpiricalSystem,validate_config
from atomic_skillgraph.empirical.budget_governor import BudgetGovernor
from atomic_skillgraph.harness.benchmarks import AnswerAdapter


def test_fixed_56_leaf_matrix():
    counts={'accepted':0,'ValueError':0}
    for field in ('decision','existing_skill_id','goal','guidance','scope_terms','applicability','rationale'):
        for value in (123,1.5,True,None,{},[],['manifold',None],'x'):
            p=deepcopy(proposal());p[field]=value
            try:validate_proposal(p,source(),[],CHOICE_PROPOSAL);counts['accepted']+=1
            except ValueError:counts['ValueError']+=1
    assert counts=={'accepted':4,'ValueError':52}


def test_six_choice_configs_and_old_config(tmp_path):
    for learning in (False,True):
        for runtime in (None,False,True):
            cfg=config(tmp_path);cfg['learning']['choice_guidance']['enabled']=learning
            if runtime is None:cfg['runtime'].pop('choice_guidance')
            else:cfg['runtime']['choice_guidance']['enabled']=runtime
            if learning and runtime is not True:
                with pytest.raises(ValueError,match='requires'):validate_config(cfg)
                assert not (tmp_path/'bank').exists()
            else:validate_config(cfg)
    assert 'choice_guidance' not in validate_config(config_for(tmp_path))['runtime']


@pytest.mark.parametrize('field,value',[('guidance',123),('existing_skill_id',[])])
def test_invalid_leaf_repairs_once_and_completed_reentry_does_not_resend(tmp_path,monkeypatch,field,value):
    bad={**proposal(),field:value}
    seen=http(monkeypatch,[response([bad],name='submit_learning'),response([{'decision':'no_change'}],name='submit_learning')])
    cfg=config(tmp_path);cfg['experiment']['output_dir']=str(tmp_path/'output')
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    try:
        result=recompile(s)
        assert result['learning']['decision']=='no_change' and len(seen)==2 and not s.bank.all('skill')
        assert recompile(s)==result and len(seen)==2
    finally:s.close()


def test_repair_exhausted_rejects_and_next_source_is_processed(tmp_path,monkeypatch):
    seen=http(monkeypatch,[response([{**proposal(),'guidance':123}],name='submit_learning'),
        response([{'decision':'no_change'}],name='submit_learning')])
    cfg=config(tmp_path);cfg['experiment']['output_dir']=str(tmp_path/'output')
    gov=BudgetGovernor(tmp_path/'budget.json',token_limit=100000,finish_reserve=0,request_limit=10,proposal_repair_limit=1)
    gov.admit('prior-repair',{'max_tokens':1},{'decision_purpose':'repair'})
    gov.complete({'request_id':'prior-repair','raw_usage':{'total_tokens':1},'outcome':'success'})
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}),budget_governor=gov)
    try:
        assert recompile(s)['learning']['decision']=='rejected'
        assert recompile(s,task('next'))['learning']['decision']=='no_change'
        assert len(seen)==2 and not s.bank.all('skill')
    finally:s.close()


def test_oserror_remains_engineering_error(tmp_path,monkeypatch):
    p=proposal();seen=http(monkeypatch,[response([p],name='submit_learning'),response([check(p)],name='submit_guidance_check')])
    cfg=config(tmp_path);cfg['experiment']['output_dir']=str(tmp_path/'output')
    s=EmpiricalSystem(cfg,harness=AnswerAdapter('livemath',{}))
    monkeypatch.setattr(s.bank,'put',lambda *a:(_ for _ in ()).throw(OSError('fixture storage failure')))
    try:
        with pytest.raises(OSError,match='storage'):recompile(s)
        assert len(seen)==2
    finally:s.close()
