"""Public final-answer formats; no parsing, scoring, or model calls."""
def answer_contract(benchmark, inputs):
    contracts = {
        'searchqa': 'Return only the requested entity or short phrase inside <answer>...</answer>. No explanation or alternative answers.',
        'docvqa': 'Return the short answer to the document question inside <answer>...</answer>. Use the supplied document images.',
        'officeqa': 'Return only the requested answer as plain text. Follow the question\'s exact format, units and precision. No answer tags, HTML escaping, explanation or alternative answers.',
    }
    if benchmark == 'livemath':
        labels = [c['label'] for c in inputs['choices']]
        return 'Return exactly one valid choice label inside <answer>...</answer>. Valid labels: ' + ', '.join(labels) + '. No explanation.'
    return contracts.get(benchmark, '')
