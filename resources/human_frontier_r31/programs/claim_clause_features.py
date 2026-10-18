"""Extract visible wording contrasts only. No mathematical theorem oracle."""
import re
MARKERS = (
  ('universal',r'\\forall|\\text\{for all\}|\bfor (?:all|every|each)\b|\ball\s+'),
  ('existential',r'\\exists|\bthere exists\b|\bexists\b'),
  ('unique',r'\bunique(?:ly)?\b|\bexactly one\b|\\exists!'),
  ('equivalence',r'\\iff|\\Leftrightarrow|\bif and only if\b|\bequivalent\b'),
  ('converse',r'\bconversely\b|\bvice versa\b'),
  ('uniform',r'\buniform(?:ly)?\b|\bglobal(?:ly)?\b'),
  ('local',r'\blocal(?:ly)?\b|\bneighborhood\b|\bnear\b'),
  ('sharp',r'\bsharp\b|\boptimal\b|\bbest possible\b'),
  ('strict_ineq',r'(?<![<>=])<(?![=<])|(?<![<>=])>(?![=>])'),
  ('weak_ineq',r'\\leq|\\geq|≤|≥|<=|>='),
  ('only_if',r'\bonly if\b'),
  ('boundary',r'\bboundary\b|\bendpoint\b|\bequality case\b'),
  ('constant_dependence',r'\bdepending on\b|\bdepends on\b|\bindependent of\b'),
)
def run(ctx, inputs):
    choices=inputs.get('choices')
    if not isinstance(choices,list) or not 2<=len(choices)<=8:
        return {'status':'needs_input','outputs':{},'detail':'Public choice array missing'}
    features=[]
    for c in choices:
        label,text=c.get('label'),c.get('text')
        if not isinstance(label,str) or not isinstance(text,str) or len(text)>16000:
            return {'status':'needs_input','outputs':{},'detail':'Invalid visible choice'}
        flags=[name for name,regex in MARKERS if re.search(regex,text,re.I)]
        features.append({'label':label,'flags':flags})
    contrasts=[]
    for i in range(len(features)):
        for j in range(i+1,len(features)):
            a,b=features[i],features[j]
            only_a=sorted(set(a['flags'])-set(b['flags']))
            only_b=sorted(set(b['flags'])-set(a['flags']))
            if only_a or only_b:
                contrasts.append({'left':a['label'],'right':b['label'],
                    'left_only':only_a,'right_only':only_b})
    return {'status':'ok','outputs':{'features':features,'contrasts':contrasts[:28],
        'scope':'surface_differences_only_no_truth_verification'}}
