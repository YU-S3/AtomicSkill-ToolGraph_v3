"""Offline helper tests only; not ASTRG production or ALFWorld tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import yaml
from experiments.prepare_r103_repeats import ALLOWED_CHANGES, changed_paths, make_configs, prepare


def template(seed=42):
    name=f'alfworld_frozen_eval_134_r103_seed{seed}'
    train=f'runs/alfworld_train_full_120_r103_seed{seed}'
    return {'repair_revision':'R10.3', 'r103_protocol_version':2,
        'data_dir':train+'/frozen/data_v3', 'trace_data_dir':'runs/'+name,
        'lifecycle':{'candidate_exploration_seed':seed},
        'llm':{'runtime':{'reasoning_effort':'high', 'max_total_tokens_per_task':600000}},
        'experiment':{'name':name,'phase':'frozen_eval','condition':'full','runtime_mode':'frozen',
            'freeze_skills':True,'seed':seed,'require_source_code_match':True,
            'require_knowledge_digest_unchanged':True,'allow_eval_traces_and_metrics':True,
            'allow_long_term_knowledge_writes':False,'resume_completed_task_boundary_only':True,
            'max_task_attempts':3,'output_dir':'runs/'+name,'task_manifest_path':'runs/'+name+'/task_manifest.json',
            'source_train_run_dir':train,'source_frozen_snapshot_dir':train+'/frozen/data_v3'}}

class PureTests(unittest.TestCase):
    def test_exact_three(self):
        self.assertEqual(len(make_configs(template(),42,Path('/repo/runs/group'))),3)
    def test_three_changes_only(self):
        base=template()
        for _,cfg in make_configs(base,42,Path('/repo/runs/group')):
            self.assertEqual(changed_paths(base,cfg),ALLOWED_CHANGES)
    def test_source_unchanged(self):
        base=template(); original=copy.deepcopy(base)
        make_configs(base,42,Path('/repo/runs/group'))
        self.assertEqual(base,original)
    def test_fixed_name_seed_source(self):
        base=template()
        for _,cfg in make_configs(base,42,Path('/repo/runs/group')):
            self.assertEqual(cfg['experiment']['name'],base['experiment']['name'])
            self.assertEqual(cfg['experiment']['seed'],42)
            self.assertEqual(cfg['data_dir'],base['data_dir'])
    def test_unique_outputs(self):
        cfgs=make_configs(template(),42,Path('/repo/runs/group'))
        self.assertEqual(len({c['experiment']['output_dir'] for _,c in cfgs}),3)
    def test_basename_and_trace(self):
        for _,cfg in make_configs(template(),42,Path('/repo/runs/group')):
            self.assertEqual(Path(cfg['trace_data_dir']).name,cfg['experiment']['name'])
            self.assertEqual(cfg['trace_data_dir'],cfg['experiment']['output_dir'])
    def test_no_mask(self):
        cfg=template(); cfg['r103_interventions']={'program_execution':False}
        with self.assertRaises(ValueError): make_configs(cfg,42,Path('/repo/runs/group'))
    def test_wrong_training_seed(self):
        with self.assertRaises(ValueError): make_configs(template(43),42,Path('/repo/runs/group'))
    def test_readonly_required(self):
        cfg=template(); cfg['experiment']['allow_long_term_knowledge_writes']=True
        with self.assertRaises(ValueError): make_configs(cfg,42,Path('/repo/runs/group'))
    def test_schema_types(self):
        cfg=template(); cfg['experiment']['seed']=42.0
        with self.assertRaises(ValueError): make_configs(cfg,42,Path('/repo/runs/group'))

class FileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
        (self.root/'configs').mkdir()
        self.cfg=template()
        (self.root/'configs/alfworld_frozen_eval_134_r103_seed42.yaml').write_text(yaml.safe_dump(self.cfg))
        self.train=self.root/self.cfg['experiment']['source_train_run_dir']
        self.bank=self.root/self.cfg['data_dir']; self.bank.mkdir(parents=True)
        self.manifest={'run_id':self.train.name,'phase':'train','tasks':[{'n':i} for i in range(120)],'code_commit':'code-hash'}
        (self.train/'run_manifest.json').write_text(json.dumps(self.manifest))
        self.freeze={'knowledge_digest':'bank-digest','provenance':{'source_run_id':self.train.name,'source_code_commit':'code-hash'}}
        (self.bank/'freeze_manifest.json').write_text(json.dumps(self.freeze))
        self.group=self.root/'runs/group'
    def tearDown(self): self.tmp.cleanup()
    def test_create(self):
        plan=prepare(self.root,42,self.group)
        self.assertEqual(plan['repeat_count'],3)
        self.assertEqual(len(list((self.group/'configs').glob('*.yaml'))),3)
    def test_bank_not_modified(self):
        before=(self.bank/'freeze_manifest.json').read_bytes()
        prepare(self.root,42,self.group)
        self.assertEqual(before,(self.bank/'freeze_manifest.json').read_bytes())
    def test_existing_group_refused(self):
        prepare(self.root,42,self.group)
        with self.assertRaises(FileExistsError): prepare(self.root,42,self.group)
    def test_missing_freeze(self):
        (self.bank/'freeze_manifest.json').unlink()
        with self.assertRaises(FileNotFoundError): prepare(self.root,42,self.group)
    def test_wrong_code(self):
        with self.assertRaises(ValueError): prepare(self.root,42,self.group,'wrong')
    def test_outside_runs_refused(self):
        with self.assertRaises(ValueError): prepare(self.root,42,self.root/'configs/repeats')
    def test_partial_manifest_refused(self):
        self.manifest['tasks']=self.manifest['tasks'][:16]
        (self.train/'run_manifest.json').write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError): prepare(self.root,42,self.group)
    def test_provenance_refused(self):
        self.freeze['provenance']['source_code_commit']='wrong'
        (self.bank/'freeze_manifest.json').write_text(json.dumps(self.freeze))
        with self.assertRaises(ValueError): prepare(self.root,42,self.group)

if __name__=='__main__':
    unittest.main(verbosity=2)
