"""Human-approved BENCHMARK-PRIOR answer routing, not theorem proof."""
def run(ctx, inputs):
    labels=inputs.get('labels',[]);meta_labels=inputs.get('meta_labels',[])
    count=inputs.get('meta_count');label=inputs.get('meta_label')
    if (type(count) is not int or count!=1 or not isinstance(label,str)
        or not isinstance(labels,list) or not isinstance(meta_labels,list)
        or len(set(labels))!=len(labels) or len(meta_labels)!=1
        or meta_labels[0]!=label or label not in labels):
        return {'status':'needs_input','outputs':{},'detail':'Not exactly one matching visible meta-option'}
    return {'status':'ok','outputs':{'answer':label,'basis':'public_benchmark_prior_not_math_proof'}}
