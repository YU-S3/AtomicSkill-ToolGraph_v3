"""Offline graph tests using the repository's real dataflow contracts."""
import importlib.util,json,sys,types,unittest
from pathlib import Path
import jsonschema
ROOT=Path(__file__).resolve().parents[1]

module_path=ROOT/'integration/native_programmatic_runtime.py'
spec=importlib.util.spec_from_file_location('r31_bridge',module_path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
BUNDLE=json.loads((ROOT/'bank.json').read_text())

class ValueStore:
 def __init__(self,task):self.task={**task.inputs,'goal':task.goal};self.results={}
 def publish(self,id,outputs,origin):self.results[id]=outputs

def validate_schema_instance(x,s):jsonschema.validate(x,s)
class Bank:
 def all(self,kind):return BUNDLE[kind]
 def get(self,key):return next((a for kind in BUNDLE['counts'] for a in BUNDLE[kind] if a['id']==key),None)
class Task:
 def __init__(self,with_meta):
  self.task_id='synthetic-only';self.physical_key='synthetic_'+str(with_meta)
  self.goal='Which statement holds?'
  self.inputs={'choices':[{'label':k,'text':('One of the remaining options is correct, but a stronger result can be proven.' if with_meta and k=='D' else 'For every x there exists a value.' if k=='A' else 'There exists a value.')} for k in 'ABCDE']}
class GraphTests(unittest.TestCase):
 def test_meta_graph_real_dataflow(self):
  ans=mod.run_programmatic_graph(Task(True),Bank(),run_worker=mod.fixture_worker,variant='benchmark_prior')
  self.assertEqual(ans['answer'],'D');self.assertEqual(ans['graph'],'workflow_lm_r31_meta_prior')
  self.assertEqual([x['node'] for x in ans['program_nodes']],['scan','prior_commit'])
  self.assertTrue(ans['program_nodes'][0]['outputs_consumed'])
  self.assertEqual(ans['worker_invocations_this_run'],2)
 def test_nonmeta_program_plus_dynamic(self):
  seen={}
  def dynamic(task,hints,node):
   seen.update(hints);return 'B',{'tokens':555,'http':1}
  ans=mod.run_programmatic_graph(Task(False),Bank(),run_worker=mod.fixture_worker,variant='benchmark_prior',dynamic_answer=dynamic)
  self.assertEqual(ans['answer'],'B');self.assertEqual([x['node'] for x in ans['program_nodes']],['scan','contrast','theorem_decide'])
  self.assertTrue(seen['contrasts'])
  self.assertEqual(ans['worker_invocations_this_run'],2)
 def test_theorem_mode_never_shortcuts_meta(self):
  ans=mod.run_programmatic_graph(Task(True),Bank(),run_worker=mod.fixture_worker,variant='theorem_reasoning')
  self.assertEqual(ans['status'],'needs_dynamic');self.assertEqual(ans['branch'],'theorem_reasoning')
  self.assertNotIn('prior_commit',[x['node'] for x in ans['program_nodes']])
if __name__=='__main__':unittest.main()
