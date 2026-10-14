"""Independent synthetic expectations for the supplied, unchanged authored code."""
from copy import deepcopy


def cases():
    rows = []
    def add(name, role, inputs, outputs, status='ok'):
        rows.append(dict(name=name, role=role, inputs=inputs, expected=outputs, expected_status=status))
    for role, subject, person, year in [('positive','Mira Vale','Ada Vale',1980),
                                       ('variation','Nora Reed','Eva Reed',1990)]:
        sentence = 'The mother of '+subject+' is '+person
        context = '[DOC]'+sentence+'.'
        doc = dict(doc_index=1,start=5,end=len(context),title=sentence+'.',text=sentence+'.')
        add('doc_catalog',role,{'context':context},{'documents':[doc],'count':1})
        add('evidence_windows',role,{'question':'mother','documents':[doc]},
            {'windows':[dict(doc_index=1,start=5,end=len(context),quote=sentence+'.',score=1,matched_terms=['mother'])],
             'covered_terms':['mother']})
        alias = 'Example Research Group (ERG)' if role=='positive' else 'Local Science Union (LSU)'
        left, right = alias[:-1].split(' (')
        add('alias_cues',role,{'context':alias},{'pairs':[dict(left=left,right=right,start=0,end=len(alias),quote=alias,cue='acronym_parenthesis')]})
        candidate = dict(answer=person,relation='parent',start=5,end=5+len(sentence),quote=sentence)
        add('explicit_relation',role,{'question':'Who is the mother of '+subject+'?','context':context},
            {'candidates':[candidate],'unique':True,'source_quote':sentence,'source_start':5})
        lines = [f'Population | {year} | 100 | people',f'Population | {year+10} | 125 | people']
        numeric = '\n'.join(lines)
        observations = []
        for i,line in enumerate(lines):
            start = 0 if i==0 else len(lines[0])+1
            observations.append(dict(subject='Population',year=year+i*10,value=100.0 if i==0 else 125.0,
                unit='people',start=start,end=start+len(line),quote=line))
        add('numeric_facts',role,{'context':numeric,'subject':'Population'},{'observations':observations})
        mode = 'latest' if role=='positive' else 'earliest'
        add('compare_facts',role,{'observations':observations,'mode':mode},
            {'selection':observations[1 if role=='positive' else 0],'mode':mode,'compared_count':2})
        add('check_supported',role,{'context':context,'quote':sentence,'start':5},
            {'exact_source_match':True,'evidence':sentence,'qualification':'lexical_source_match_not_entailment'})
        add('render_relation_answer',role,{'candidates':[candidate],'verified':True},{'answer':person})
    add('doc_catalog','invalid',{'context':''},{},'needs_input')
    doc = next(r['expected']['documents'][0] for r in rows if r['name']=='doc_catalog')
    add('evidence_windows','invalid',{'question':'','documents':[doc]},{},'needs_input')
    add('alias_cues','invalid',{'context':'Only a word without an alias.'},{},'not_found')
    add('explicit_relation','invalid',{'question':'Who is the mother of Mira Vale?',
        'context':'The mother of Mira Vale is Ada Vale. The mother of Mira Vale is Eva Reed.'},{},'needs_input')
    add('numeric_facts','invalid',{'context':'Population | 1980 | 100 | people','subject':'Absent'},{},'not_found')
    mixed = deepcopy(next(r['inputs']['observations'] for r in rows if r['name']=='compare_facts'))
    mixed[-1]['unit']='million'
    add('compare_facts','invalid',{'observations':mixed,'mode':'latest'},{},'needs_input')
    add('check_supported','invalid',{'context':'abcdef','quote':'bcd','start':0},
        {'exact_source_match':False,'evidence':'','qualification':'lexical_source_match_not_entailment'})
    candidate = deepcopy(next(r['inputs']['candidates'] for r in rows if r['name']=='render_relation_answer'))
    add('render_relation_answer','invalid',{'candidates':candidate,'verified':False},{},'needs_input')
    return rows
