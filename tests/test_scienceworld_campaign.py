import json
from pathlib import Path
import pytest
from experiments import scienceworld_campaign as campaign


@pytest.mark.parametrize('seed,repeats',[(42,3),(43,1),(44,1)])
def test_lane_orders_compile_readonly_dev_and_same_bank_test(tmp_path,monkeypatch,seed,repeats):
    configs,finished,events={},{},[]
    def make(root,*,seed,phase='train',repetition=1,frozen=None,**kw):
        name=f'{phase}_{repetition}'
        path=tmp_path/(name+'.yaml')
        configs[path]={'data_dir':str(frozen or tmp_path/f'seed{seed}/train/data_v3'),
            'experiment':{'phase':phase,'output_dir':str(tmp_path/f'seed{seed}'/name)}}
        return path
    monkeypatch.setattr(campaign,'make_config',make)
    monkeypatch.setattr(campaign,'load_config',lambda p:configs[Path(p)])
    monkeypatch.setattr(campaign,'completed',lambda p:finished.get(p,False))
    monkeypatch.setattr(campaign,'bank_digest',lambda p:'compiled' if 'compiled' in str(p) else 'raw')
    def compile(config,output):
        events.append('compile');output.mkdir()
        report={'completed':True,'source_train_bank_digest':'raw','compiled_bank_digest':'compiled'}
        (output/'compiler_manifest.json').write_text(json.dumps(report))
        return report
    monkeypatch.setattr(campaign,'compile_train',compile)
    def call(command,**kw):
        path=Path(command[command.index('--config')+1]);cfg=configs[path]
        events.append(cfg['experiment']['phase'])
        if cfg['experiment']['phase']!='train':
            assert cfg['data_dir']==str(tmp_path/f'seed{seed}/compiled/data_v3')
        finished[path]=True
        return 0
    monkeypatch.setattr(campaign.subprocess,'call',call)
    assert campaign.run_lane(tmp_path,seed)==0
    assert events==['train','compile','dev']+['test']*repeats
    assert campaign.run_lane(tmp_path,seed,resume=True)==0
    assert len(events)==3+repeats
    with pytest.raises(FileExistsError):campaign.run_lane(tmp_path,seed)


def test_stop_file_stops_lane_before_next_phase_without_killing_or_writes(tmp_path,monkeypatch):
    lane=tmp_path/'seed42';lane.mkdir();(lane/'STOP_AFTER_TASK').touch()
    monkeypatch.setattr(campaign,'make_config',lambda *a,**k:tmp_path/'config.yaml')
    monkeypatch.setattr(campaign.subprocess,'call',lambda *a,**k:pytest.fail('Stopped lane must not launch'))
    assert campaign.run_lane(tmp_path,42)==75


@pytest.mark.parametrize('seed,repeats',[(42,3),(43,1),(44,1)])
def test_authored_launcher_is_readonly_test_only_and_preserves_digest(tmp_path,monkeypatch,seed,repeats):
    from experiments import scienceworld_reference_test as reference
    bank=tmp_path/'authored';bank.mkdir()
    (bank/'freeze_manifest.json').write_text(json.dumps({'experiment_kind':'authored_reference','knowledge_digest':'fixed'}))
    configs,finished,calls={},{},[]
    def make(root,**kw):
        assert kw['authored_reference'] and kw['phase']=='test' and kw['frozen']==bank
        path=tmp_path/f"repeat{kw['repetition']}.yaml";configs[path]=kw
        return path
    monkeypatch.setattr(reference,'make_config',make)
    monkeypatch.setattr(reference,'bank_digest',lambda p:'fixed')
    monkeypatch.setattr(reference,'completed',lambda p:finished.get(p,False))
    def call(cmd,**kw):
        path=Path(cmd[cmd.index('--config')+1]);calls.append(configs[path]);finished[path]=True
        return 0
    monkeypatch.setattr(reference.subprocess,'call',call)
    assert reference.run(tmp_path/'runs',bank,seed)==0
    assert len(calls)==repeats
    assert reference.run(tmp_path/'runs',bank,seed,resume=True)==0
    assert len(calls)==repeats


def test_reference_launcher_rejects_learned_snapshot(tmp_path):
    from experiments import scienceworld_reference_test as reference
    bank=tmp_path/'learned';bank.mkdir()
    (bank/'freeze_manifest.json').write_text(json.dumps({'knowledge_digest':'x'}))
    with pytest.raises(ValueError,match='authored reference'):
        reference.run(tmp_path/'runs',bank,42)
