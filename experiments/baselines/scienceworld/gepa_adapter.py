"""GEPA integration; preserves its optimizer, Pareto and acceptance semantics."""
import json
from pathlib import Path
import uuid
from gepa.core.adapter import EvaluationBatch
from .runner import ScienceWorldTextEpisodeRunner

class ScienceWorldGEPAAdapter:
    def __init__(self, output, chat_factory, *, max_steps=100):
        self.output, self.chat_factory, self.max_steps = Path(output), chat_factory, max_steps
        self.propose_new_texts = None  # Use GEPA's original reflection proposer.

    def evaluate(self, batch, candidate, capture_traces=False):
        if set(candidate) != {'skill_text'}:
            raise ValueError('Only skill_text may evolve')
        scope = self.output / ('evaluation_' + uuid.uuid4().hex)
        results, trajectories = [], []
        for entry in batch:
            episode = scope / entry['task_id']
            chat = self.chat_factory(episode, entry)
            result = ScienceWorldTextEpisodeRunner(chat, max_actions=self.max_steps).run(entry, candidate['skill_text'], episode)
            results.append(result)
            if capture_traces:
                trajectories.append({'task': entry, 'result': result,
                    'conversation': json.loads((episode / 'conversation.json').read_text())})
        return EvaluationBatch(outputs=results, scores=[r['normalized_score'] for r in results],
                               trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if components_to_update != ['skill_text'] or eval_batch.trajectories is None:
            raise ValueError('Reflection requires captured skill_text executions')
        return {'skill_text': [{'Inputs': t['task'], 'Generated Outputs': t['conversation'],
                'Feedback': {'official_normalized_score': s, 'perfect_success': t['result']['perfect_success'],
                             'termination_reason': t['result']['termination_reason']}}
            for t, s in zip(eval_batch.trajectories, eval_batch.scores)]}
