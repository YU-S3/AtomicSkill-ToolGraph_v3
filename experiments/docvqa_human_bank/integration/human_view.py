"""Readonly manual-only Bank adapter for real DocVQA AnswerExecutor.

Uses the production single_answer provider, original inline public image, scorer,
checkpoint and ProgramWorker. Never modifies general qualification rules.
"""
import base64
from contextlib import contextmanager
from copy import deepcopy
import hashlib
from io import BytesIO
import re
from pathlib import Path

from PIL import Image

METHOD='docvqa.human-ideal.programmatic.v1.1'


def skill_selection(question):
    """Cheap public-question routing; no model call or hidden answers."""
    q=question.lower()
    extra=[]
    patterns=[
      ('DV03',r'\b(table|column|row|across|percentage of|figure in|net|gross|total on)\b'),
      ('DV02',r'\b(form|invoice|receipt|issued to|address|telephone|phone|name of|signature|signed by|account)\b'),
      ('DV04',r'\b(page number|contents|agenda|item number|section|chapter|bullet|list|which point)\b'),
      ('DV05',r'\b(handwrit|written|scribbl|signature|serial|id number|registration|number on|code)\b'),
      ('DV06',r'\b(how many|how much|how long|date|year|amount|percent|\%|currency|price|number|time)\b'),
      ('DV07',r'\b(diagram|graph|chart|drawn|arrow|checkbox|checked|circle|tick|figure|symbol|mark)\b'),
      ('DV08',r'\b(total|sum|difference|average|ratio|growth|increase|decrease|between|combined)\b'),
      ('DV09',r'\b(exact|quote|word|phrase|title|written|label|called|name)\b'),
    ]
    for key,pattern in patterns:
        if re.search(pattern,q): extra.append(key)
    # Short, verbatim answers apply regardless of question wording.
    return ['DV01', 'DV10', *extra[:2]]


class HumanDocVQAView:
    """Manual evaluation view over a native Bank; Program calls are not Ours learning claims."""
    def __init__(self,bank,manifest,*,arm='human',programs='auto'):
        if not bank.readonly: raise ValueError('Readonly Bank required')
        if arm not in ('human','off') or programs not in ('off','auto','all'):
            raise ValueError('Unrecognized arm or program mode')
        from atomic_skillgraph.empirical.contracts import digest
        self.bank,self.arm,self.program_mode=bank,arm,programs
        self.manifest=deepcopy(manifest)
        if manifest['method_version']!=METHOD or manifest['bank_digest']!=bank.digest():
            raise ValueError('Native Bank identity differs from manual manifest')
        self.skills={s['title']:s for s in bank.all('skill')}
        self.programs={Path(p['source_file']).stem:p for p in bank.all('program')}
        self.allowed_ids={p['id'] for p in self.programs.values()}
        self.view_hash=digest([METHOD,manifest,arm,programs])

    def __getattr__(self,key):return getattr(self.bank,key)
    def get(self,key):return self.bank.get(key) if self.arm=='human' else None
    def all(self,kind):return self.bank.all(kind) if self.arm=='human' else []
    def selected(self,question,limit=5):
        if self.arm=='off':return []
        return [self.skills[k] for k in skill_selection(question) if k in self.skills][:limit]
    def retrieve_guidance(self,query,limit=5):return self.selected(query,limit)
    def retrieve(self,query,limit=8):return self.selected(query,limit)
    def guidance_retrieval_audit(self,query,assets):
        return [{'rank':i,'skill_id':s['id'],'manual_source':'human_ideal_reference'} for i,s in enumerate(assets)]
    def planning_cards(self,query):return []  # single-answer task; no unused planner turn
    def workflow_options(self):return self.all('workflow')
    def routes(self,node,**kwargs):return [] # no automatic speculative program chain
    def program_eligible(self,program):
        return bool(self.arm=='human' and self.program_mode!='off' and program
                    and program['id'] in self.allowed_ids
                    and self.bank.get(program['id'])==program
                    and program['environment']==self.bank.environment
                    and program.get('state')=='candidate')
    def program_options(self,query,*,excluded=(),limit=4,**kwargs):
        if self.arm!='human' or self.program_mode=='off':return []
        q=query.lower()
        # Image legibility is visible in the image, not inferable from question keywords.
        names=['image_region']
        if self.program_mode=='all' or re.search(r'\b(sum|total|average|mean|difference|ratio|increase|decrease|percent|count)\b',q):
            names.append('numeric_compute')
        if self.program_mode=='all' or re.search(r'\b(phrase|written|title|quote|label|form)\b',q):
            names.append('transcription_check')
        return [{'id':p['id'],'state':'callable_manual',
                 'goal':p['goal'],'result_role':p['result_role'],
                 'input_schema':p['input_schema'],'output_schema':p['output_schema'],
                 'entry_constraints':p.get('entry_constraints',{}),
                 'manual_authority':METHOD}
               for k in names if (p:=self.programs.get(k)) is not None
               and p['id'] not in excluded and self.program_eligible(p)][:limit]


def safe_crop_part(context,task,bank):
    """After a *real* Program result, return one verified crop for an image user message."""
    if context is None:return None
    try:
        from atomic_skillgraph.empirical.public_results import PROGRAM_EXECUTION
    except ImportError:return None
    sources={hashlib.sha256(Path(p).read_bytes()).hexdigest()
             for p in task.inputs.get('images',[]) if Path(p).is_file()}
    crop=bank.programs.get('image_region')
    if not crop or not bank.program_eligible(crop):return None
    produced={r['result_id'] for r in context.memory.values() if r['name']==crop['id']}
    for rid,record in reversed(list(context.results.items())):
        try:
            if rid not in produced or not record.get('execution_receipt'):continue
            if context.origin(rid)!=PROGRAM_EXECUTION or record.get('status')!='ok':continue
            out=record.get('outputs',{})
            if not all(k in out for k in ('crop_base64','image_mime','image_sha256','source_sha256')):continue
            if out['image_mime']!='image/jpeg' or out['source_sha256'] not in sources:continue
            raw=base64.b64decode(out['crop_base64'],validate=True)
            if len(raw)>1_000_000 or hashlib.sha256(raw).hexdigest()!=out['image_sha256']:continue
            with Image.open(BytesIO(raw)) as im:
                if im.format!='JPEG' or max(im.size)>1600:continue
                im.verify()
            return {'result_id':rid,'image_sha256':out['image_sha256'],
                    'bbox':out['bbox_normalized'],
                    'part':{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+out['crop_base64'],'detail':'high'}}}
        except (ValueError,KeyError,TypeError,OSError):continue
    return None


def scrub_crop_binary(value):
    if isinstance(value,dict):
        return {k:('[inline crop rendered as an image; use crop_visual_evidence result_id]'
                    if k=='crop_base64' else scrub_crop_binary(v)) for k,v in value.items()}
    if isinstance(value,list):return [scrub_crop_binary(x) for x in value]
    return value


@contextmanager
def image_result_bridge():
    """Only the manual DocVQA view receives additional images after actual Crop Worker output."""
    from atomic_skillgraph.empirical.system import EmpiricalSystem
    old=EmpiricalSystem.agent
    def agent(s,stage,prompt,materials,name,schema,**kwargs):
        if stage=='runtime' and isinstance(s.bank,HumanDocVQAView) and s.bank.arm=='human':
            prompt=prompt.replace('supplied usable ID','supplied manually authorized reference ID')
            prompt+='\nHuman-authored DocVQA reference arm: these assets are guidance, not facts. Answer directly when the document is clear. Programs are optional. call_program requires name=the exact displayed ID and arguments matching that Program input_schema. image_region returns a real crop visible in the next request alongside the original image; never read base64 as text. numeric_compute requires already observed operands in the documented order. A transcription match only checks your own text, not the image. After a failed or unhelpful tool result, answer from the image rather than repeat the same operation. No extra explanation in the final answer.'
            v=deepcopy(materials)
            v['content_parts']=deepcopy(v.get('content_parts') or s.adapter.content_parts())
            entry=safe_crop_part(s.task_context,s.adapter.task,s.bank)
            # Respect the actual provider lock. No extra image if the Provider cannot take it.
            if entry and 'image' in s.config.get('llm',{}).get('input_modalities',[]):
                maximum=s.config.get('llm',{}).get('image_input',{}).get('max_images',1)
                if len(v['content_parts'])<maximum:
                    v['content_parts'].append(entry['part'])
                    v['crop_visual_evidence']={'result_id':entry['result_id'],'image_sha256':entry['image_sha256'],
                                               'bbox_normalized':entry['bbox'],
                                               'note':'This is a crop of the original document, not separate proof'}
            for key in tuple(v):
                if key!='content_parts':v[key]=scrub_crop_binary(v[key])
            if v.get('remaining_local_executions')==0:
                v['programs']=[]
                prompt+=' No local executions remain. Submit the final answer now using the available image evidence.'
            materials=v
        return old(s,stage,prompt,materials,name,schema,**kwargs)
    EmpiricalSystem.agent=agent
    try:yield
    finally:EmpiricalSystem.agent=old
