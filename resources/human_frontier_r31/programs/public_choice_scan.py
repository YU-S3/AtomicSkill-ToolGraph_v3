"""Extract the only public stronger-result meta-choice; never access model or gold."""
import re
META = re.compile(r'(?:one of (?:the )?(?:remaining|other) options is correct.{0,120}(?:stronger result|stronger statement|stronger conclusion) can be proven)|(?:none of the above.{0,80}strongest)', re.I | re.S)

def run(ctx, inputs):
    choices=inputs.get('choices',[])
    if not isinstance(choices,list) or not 2<=len(choices)<=8:
        return {'status':'needs_input','outputs':{},'detail':'2..8 real public choices required'}
    labels=[]; matches=[]; raw=[]
    for c in choices:
        if not isinstance(c,dict) or not isinstance(c.get('text'),str) or not isinstance(c.get('label'),str):
            return {'status':'needs_input','outputs':{},'detail':'Each choice needs public label/text'}
        label,text=c['label'],c['text']
        if label not in {'A','B','C','D','E','F','G','H'} or label in labels or len(text)>16000:
            return {'status':'needs_input','outputs':{},'detail':'Invalid/repeated label or oversized choice'}
        labels.append(label)
        if META.search(text):matches.append(label)
        raw.append({'label':label,'text':text})
    return {'status':'ok','outputs':{
        'labels':labels,'choices':raw,'meta_labels':matches,'meta_count':len(matches),
        'meta_label':matches[0] if len(matches)==1 else '',
        'scope':'public_choices_only_not_theorem_proof'}}
