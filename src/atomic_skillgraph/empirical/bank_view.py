"""Readonly diagnostic interventions without changing the underlying Bank."""
from .contracts import digest


class BankView:
    version = 'empirical.bank-view.v1'
    hidden = {'all', 'retrieve', 'retrieve_guidance', 'planning_cards', 'routes',
              'program_options', 'attempts', 'jobs', 'train_cases'}
    mutators = {'put', 'record', 'save_job', 'save_case', 'quarantine_program', 'freeze', 'update_choice_statistics'}

    def __init__(self, bank, mode):
        if not bank.readonly or mode not in {'guidance_off', 'learned_assets_off'}:
            raise ValueError('Diagnostic views require a readonly Bank and explicit mode')
        self.bank, self.mode = bank, mode
        self.view_hash = digest([self.version, mode])

    def select_guidance(self, task, policy):
        # Still validate the Frozen statistics identity before suppressing exposure.
        selection = self.bank.select_guidance(task, policy)
        return {**selection, 'selected': [], 'injected_ids': [], 'injected_chars': 0,
                'view': self.mode, 'view_hash': self.view_hash,
                'audit': [{**row, 'reason': 'view_hidden'} for row in selection['audit']]}

    def __getattr__(self, name):
        if name in self.mutators:
            def refuse(*args, **kwargs): raise RuntimeError('Diagnostic Bank view is readonly')
            return refuse
        if self.mode == 'learned_assets_off' and name == 'get': return lambda *a, **k: None
        if (self.mode == 'learned_assets_off' and name in self.hidden) or name == 'retrieve_guidance':
            return lambda *a, **k: []
        return getattr(self.bank, name)
