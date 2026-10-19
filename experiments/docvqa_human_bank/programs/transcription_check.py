"""Bounded check against model-provided transcription, not against image truth."""
import re


def run(ctx,inputs):
    try:
        line=inputs['observed_text']
        candidate=inputs['candidate']
        anchor=inputs.get('anchor','')
        if not isinstance(line,str) or not isinstance(candidate,str) or not isinstance(anchor,str):
            raise ValueError('All inputs are strings')
        if not 0<len(candidate)<=300 or len(line)>12000 or len(anchor)>300:
            raise ValueError('Invalid text length')
        norm=lambda s:' '.join(s.split()).casefold()
        match=norm(candidate) in norm(line)
        anchored=not anchor or norm(anchor) in norm(line)
        raw_exact=candidate in line
        return {'status':'ok','outputs':{'candidate_present':match,
            'anchor_present':anchored,'verbatim_present':raw_exact,
            'candidate':candidate,
            'scope':'model_supplied_transcription_only_not_visual_source_certification'}}
    except (KeyError,ValueError,TypeError) as exc:
        return {'status':'needs_input','outputs':{},'detail':str(exc)[:400]}
