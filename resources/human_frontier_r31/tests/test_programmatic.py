"""Fully offline fixture contract checks; NEVER represent as paid benchmark success."""
import importlib.util,json,sqlite3,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import jsonschema
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('r31_graph',ROOT/'integration/native_programmatic_runtime.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
BANK=json.loads((ROOT/'bank.json').read_text())
class InMemoryBank:
 def all(self,k):return BANK[k]
 def get(self,key):return next((x for k in ('skill','program','implementation','workflow') for x in BANK[k] if x['id']==key),None)

class Task:
 def __init__(self,choices):
  self.goal='Which is the strongest provable statement?';self.inputs={'choices':choices}
  self.physical_key='offline_fixture';self.task_id='synthetic'

def c(label,text):return {'label':label,'text':text}
class Tests(unittest.TestCase):
 def test_program_digest_matches_native_algorithm(self):
  for p in BANK['program']:
   ks=['source','entry','input_schema','output_schema','allowed_tools','environment','result_role','entry_constraints']
   self.assertEqual(p['id'],'program_'+mod.digest({k:p[k] for k in ks}))
 def test_type_contracts(self):
  for p in BANK['program']:
   self.assertEqual(p['entry'],'run');self.assertIn('image_digest',p['environment'])
   self.assertEqual(p['allowed_tools'],[])
   compile(p['source'],p['source_file'],'exec')
 def test_structural_graph_dataflow(self):
  id_programs={p['id']:p for p in BANK['program']}
  links={(i['skill_id'],i['program_id']) for i in BANK['implementation']}
  self.assertEqual(len(links),3)
  for wf in BANK['workflow']:
   seen={}
   for node in wf['nodes']:
    for k,ref in node.get('args',{}).items():
     if 'from' in ref:
      self.assertIn(ref['from'],seen)
      self.assertIn(ref['field'],seen[ref['from']])
    if node['execution_mode']=='program':
     p=id_programs[node['program_id']];seen[node['id']]=set(p['output_schema']['properties'])
    else: seen[node['id']]={'answer'}
   self.assertEqual(wf['outputs']['answer']['field'],'answer')
 def test_matching_meta_any_label_and_no_golds(self):
  for label in 'ABCDE':
   task=Task([c(l,'One of the remaining options is correct, but a stronger result can be proven.' if l==label else 'A mathematical claim '+l) for l in 'ABCDE'])
   # Native program code exercised with exactly the JSON it would receive.
   scan=BANK['program'][0];a=mod.fixture_worker(scan,{'choices':task.inputs['choices']})
   self.assertEqual(a['status'],'ok');jsonschema.validate(a['outputs'],scan['output_schema'])
   prior=BANK['program'][1];b=mod.fixture_worker(prior,{k:a['outputs'][k] for k in ('labels','meta_labels','meta_count','meta_label')})
   self.assertEqual(b['outputs']['answer'],label)
 def test_ambiguous_prior_fails_closed(self):
  c1=[c('A','One of the remaining options is correct, but a stronger result can be proven.'),
      c('B','One of the remaining options is correct, but a stronger result can be proven.'),
      c('C','Another choice')]
  a=mod.fixture_worker(BANK['program'][0],{'choices':c1})
  self.assertEqual(a['outputs']['meta_count'],2)
  b=mod.fixture_worker(BANK['program'][1],{k:a['outputs'][k] for k in ('labels','meta_labels','meta_count','meta_label')})
  self.assertEqual(b['status'],'needs_input')
 def test_nonmeta_contrast_is_not_mathematical_proof(self):
  x=[c('A','There exists a unique solution.'),c('B','For every x, there exists a solution.'),c('C','There is an equivalence if and only if the condition holds.')]
  a=mod.fixture_worker(BANK['program'][2],{'choices':x})
  self.assertEqual(a['status'],'ok');self.assertIn('no_truth_verification',a['outputs']['scope'])
 def test_source_db_integrity(self):
  cx=sqlite3.connect(ROOT/'bank.sqlite3')
  rows=dict(cx.execute('SELECT kind,count(*) FROM assets GROUP BY kind'))
  self.assertEqual(rows,BANK['counts']);cx.close()

if __name__=='__main__':unittest.main()
