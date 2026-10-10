"""Readonly diagnostic interventions without changing the underlying Bank."""
from .contracts import digest


class BankView:
    version = 'empirical.bank-view.v2'
    hidden = {'all', 'retrieve', 'retrieve_guidance', 'planning_cards', 'routes',
              'program_options', 'attempts', 'jobs', 'train_cases'}
    mutators = {'put', 'record', 'record_validation', 'save_job', 'save_case', 'quarantine_program', 'freeze', 'update_choice_statistics'}

    def __init__(self, bank, mode):
        if not bank.readonly or mode not in {'guidance_off', 'guidance_only', 'learned_assets_off'}:
            raise ValueError('Diagnostic views require a readonly Bank and explicit mode')
        self.bank, self.mode = bank, mode
        self.view_hash = digest([self.version, mode])

    def program_eligible(self, program):
        return bool(program and self.mode == 'guidance_off' and self.bank.program_eligible(self.get(program['id'])))

    def all(self, kind):
        if self.mode == 'learned_assets_off' or self.mode == 'guidance_only' and kind != 'skill': return []
        assets = self.bank.all(kind)
        if self.mode == 'guidance_only':
            return [{k:a[k] for k in ('id','goal','guidance','execution_intent','evidence_source','grounding_check',
                'grounded_proposal','guidance_policy_version','scope_terms','applicability','parent_skill_id') if k in a}
                for a in assets if isinstance(a.get('guidance'), str) and a['guidance'].strip()]
        return assets

    def get(self, asset_id):
        return next((a for kind in ('skill','program','implementation','workflow') for a in self.all(kind) if a['id']==asset_id), None)

    def retrieve(self, query, limit=8):
        if self.mode != 'guidance_only': return [] if self.mode == 'learned_assets_off' else self.bank.retrieve(query, limit)
        return sorted(self.all('skill'), key=lambda a:(-len(self.bank.words(query)&self.bank.words(a.get('goal',''))),a['id']))[:limit]

    def retrieve_guidance(self, query, limit=8):
        if self.mode != 'guidance_only': return []
        visible = {a['id']:a for a in self.all('skill')}
        return [visible[a['id']] for a in self.bank.retrieve_guidance(query,limit) if a['id'] in visible]

    def select_guidance(self, task, policy):
        # Still validate the Frozen statistics identity before suppressing exposure.
        selection = self.bank.select_guidance(task, policy)
        if self.mode == 'guidance_only': return selection
        return {**selection, 'selected': [], 'injected_ids': [], 'injected_chars': 0,
                'view': self.mode, 'view_hash': self.view_hash,
                'audit': []}

    def __getattr__(self, name):
        if name in self.mutators:
            def refuse(*args, **kwargs): raise RuntimeError('Diagnostic Bank view is readonly')
            return refuse
        if self.mode == 'learned_assets_off' and name == 'get': return lambda *a, **k: None
        if self.mode in {'learned_assets_off','guidance_only'} and name in self.hidden:
            return lambda *a, **k: []
        return getattr(self.bank, name)
