"""One version state and one attempt table; no per-layer qualification."""
import json
import sqlite3
import re
from pathlib import Path

from .contracts import digest, program_digest, validate_program, validate_workflow, resolve_node_interface, normalize_workflow


class Bank:
    def __init__(self, root, *, readonly=False, seed=42):
        self.root, self.readonly = Path(root), readonly
        self.observer = None
        if readonly:
            self.db = sqlite3.connect(f"{(self.root / 'bank.sqlite3').resolve().as_uri()}?mode=ro", uri=True)
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            self.db = sqlite3.connect(self.root / "bank.sqlite3")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS assets(kind TEXT,id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY,program_id TEXT,task_key TEXT,
                    origin TEXT,outcome TEXT,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS realization_jobs(id TEXT PRIMARY KEY,payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS train_cases(task_key TEXT PRIMARY KEY,payload TEXT NOT NULL);
            """)
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('schema','empirical.bank.v1')")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('seed',?)", (str(seed),))
            self.db.commit()
        schema = self.db.execute("SELECT value FROM metadata WHERE key='schema'").fetchone()
        stored_seed = self.db.execute("SELECT value FROM metadata WHERE key='seed'").fetchone()
        if schema != ("empirical.bank.v1",) or stored_seed != (str(seed),):
            self.db.close()
            raise ValueError("Bank schema/seed mismatch; legacy Banks cannot be migrated")

    def _writable(self):
        if self.readonly:
            raise RuntimeError("Frozen Bank is read-only")

    def put(self, kind, asset):
        self._writable()
        asset = dict(asset)
        if kind == "program":
            if not isinstance(asset.get('allowed_tools'), list):
                raise ValueError('allowed_tools must be tool names')
            asset["allowed_tools"] = sorted(set(asset["allowed_tools"]))
            asset["id"] = "program_" + program_digest(asset)
            existing = self.get(asset['id'])
            if existing is not None:
                return existing
            validate_program(asset)
            asset["state"] = "candidate"
        elif kind == "workflow":
            asset = normalize_workflow(asset, self)
            validate_workflow(asset, bank=self)
        if kind not in {"skill", "implementation", "program", "workflow"}:
            raise ValueError("Unknown asset kind")
        asset.setdefault("id", kind + "_" + digest(asset))
        old = self.get(asset["id"])
        if old is not None:
            comparable = {k: v for k, v in old.items() if k != "state"}
            proposed = {k: v for k, v in asset.items() if k != "state"}
            if kind != "program" and comparable != proposed:
                raise ValueError("Asset id content conflict; create a new version")
            return old
        self.db.execute("INSERT INTO assets VALUES(?,?,?)", (kind, asset["id"], json.dumps(asset, ensure_ascii=False)))
        self.db.commit()
        if self.observer:
            self.observer.asset(kind, asset)
        return asset

    def get(self, asset_id):
        row = self.db.execute("SELECT payload FROM assets WHERE id=?", (asset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM assets WHERE kind=? ORDER BY id", (kind,))]

    def routes(self, node, *, allow_candidate=False):
        interface = resolve_node_interface(node, self)
        if interface['execution_mode'] == 'dynamic':
            return []
        refs = [interface['bound_program_id']] if interface['execution_mode'] == 'program' else [
            i["program_id"] for i in self.all("implementation") if i["skill_id"] == interface['bound_skill_id']]
        routes = [self.get(ref) for ref in refs]
        routes = [p for p in routes if p and (p["state"] == "usable" or (
            allow_candidate and not self.readonly and p["state"] == "candidate"))]
        def order(program):
            rows = self.attempts(program["id"])
            valid = [a for a in rows if a["outcome"] in {"positive", "execution_failure"}]
            successes = sum(a["outcome"] == "positive" for a in valid)
            costs = [a["calls"] for a in valid if "calls" in a]
            return (program["state"] != "usable", -(successes / len(valid) if valid else 0),
                    sum(costs) / len(costs) if costs else float("inf"), program["id"])
        return sorted(routes, key=order)

    def attempts(self, program_id):
        return [json.loads(r[0]) for r in self.db.execute(
            "SELECT payload FROM attempts WHERE program_id=? ORDER BY rowid", (program_id,))]

    def record(self, attempt):
        self._writable()
        for key in ("id", "program_id", "task_key", "origin", "outcome"):
            if key not in attempt:
                raise ValueError("Attempt missing " + key)
        if attempt["outcome"] == "positive" and attempt.get("basis") not in {"local_check", "task_outcome"}:
            raise ValueError("Positive trial requires independent local or task evaluation")
        old = self.db.execute("SELECT payload FROM attempts WHERE id=?", (attempt["id"],)).fetchone()
        payload = json.dumps(attempt, ensure_ascii=False, sort_keys=True)
        if old:
            if json.loads(old[0]) != attempt:
                raise ValueError("Attempt id conflict")
            return
        program = self.get(attempt["program_id"])
        if program is None:
            raise ValueError("Attempt references unknown program")
        with self.db:
            self.db.execute("INSERT INTO attempts VALUES(?,?,?,?,?,?)", (
                attempt["id"], attempt["program_id"], attempt["task_key"], attempt["origin"], attempt["outcome"], payload))
            rows = self.attempts(program["id"])
            positives = {a["task_key"] for a in rows if a["outcome"] == "positive"}
            if len(positives) >= 2 and program["state"] != "disabled":
                program["state"] = "usable"
            failures = 0
            for row in reversed(rows):
                if row["outcome"] == "positive":
                    break
                if row["outcome"] == "execution_failure":
                    failures += 1
                elif row["outcome"] == "normal":
                    break
            if failures >= 2:
                program["state"] = "disabled"
            self.db.execute("UPDATE assets SET payload=? WHERE id=?", (json.dumps(program), program["id"]))
        if self.observer:
            self.observer.asset('program', program)

    def retrieve(self, query, limit=8):
        words = self.words(query)
        assets = [*self.all("skill"), *self.all("workflow")]
        return sorted(assets, key=lambda a: (-len(words & self.words(a.get('goal', '') + ' ' + a.get('guidance', ''))), a['id']))[:limit]

    @staticmethod
    def words(text): return set(re.findall(r'\w+', text.casefold(), flags=re.UNICODE))

    def retrieve_guidance(self, query, limit=8):
        assets = [a for a in self.all('skill') if a.get('execution_intent') == 'guidance_only'
                  and isinstance(a.get('guidance'), str) and a['guidance'].strip()]
        superseded = {a['parent_skill_id'] for a in assets if a.get('parent_skill_id')}
        words = self.words(query)
        return sorted((a for a in assets if a['id'] not in superseded), key=lambda a: (
            -len(words & self.words(a.get('goal', '') + ' ' + a['guidance'])), a['id']))[:limit]

    def planning_cards(self, query):
        cards = []
        for asset in self.retrieve(query):
            if 'guidance' in asset or 'input_schema' in asset:
                card = {k: asset[k] for k in ('id', 'goal', 'guidance', 'input_schema', 'output_schema',
                        'execution_intent', 'result_role', 'entry_constraints') if k in asset}
                card['usable_program_ids'] = [p['id'] for p in self.routes({'execution_mode': 'skill', 'skill_id': asset['id']})]
            else:
                card = {k: asset[k] for k in ('id', 'goal', 'nodes', 'outputs', 'interface_version') if k in asset}
                summary = dict.fromkeys(('dynamic_nodes', 'bound_skill_nodes_with_usable_program',
                    'bound_skill_nodes_without_usable_program', 'explicit_usable_program_nodes'), 0)
                ambiguous = []
                for node in asset['nodes']:
                    if not node.get('execution_mode') and node.get('skill_id') and not node.get('program_id'):
                        ambiguous.append(node['id'])
                        continue
                    mode = resolve_node_interface(node, self)['execution_mode']
                    if mode == 'dynamic': key = 'dynamic_nodes'
                    elif mode == 'skill': key = 'bound_skill_nodes_with_usable_program' if self.routes(node) else 'bound_skill_nodes_without_usable_program'
                    else:
                        if not self.routes(node): continue
                        key = 'explicit_usable_program_nodes'
                    summary[key] += 1
                card['execution_summary'] = summary
                if ambiguous: card['requires_explicit_node_modes'] = ambiguous
            cards.append(card)
        return cards

    def program_options(self, query, *, node=None, allow_candidate=False, excluded=(), limit=8):
        interface = resolve_node_interface(node, self) if node else None
        authorized = {p['id'] for p in self.routes(node, allow_candidate=allow_candidate)} if node else set()
        skills = {s['id']: s for s in self.all('skill')}
        cards = []
        for p in self.all('program'):
            if p['id'] in excluded or p['state'] == 'disabled' or (p['state'] != 'usable' and not (allow_candidate and not self.readonly)):
                continue
            capabilities = [{'skill_id': i['skill_id'], 'goal': skills.get(i['skill_id'], {}).get('goal', '')}
                            for i in self.all('implementation') if i['program_id'] == p['id']]
            description = ' '.join(c['goal'] for c in capabilities)
            trials = [a for a in self.attempts(p['id']) if a['outcome'] in {'positive','execution_failure'}]
            reliability = sum(a['outcome']=='positive' for a in trials)/len(trials) if trials else 0
            card = {'id': p['id'], 'state': p['state'], 'capabilities': capabilities,
                    'entry_constraints': p.get('entry_constraints', 'undeclared'),
                    'result_role': p.get('result_role', 'intermediate'),
                    'input_schema': p['input_schema'], 'output_schema': p['output_schema']}
            order = (p['id'] not in authorized, -len(self.words((interface or {}).get('node_goal','')) & self.words(description)),
                     -len(self.words(query) & self.words(description)), p['state'] != 'usable', -reliability, p['id'])
            cards.append((order, card))
        return [card for _, card in sorted(cards, key=lambda row: row[0])[:limit]]

    def jobs(self):
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='realization_jobs'").fetchone(): return []
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM realization_jobs ORDER BY rowid')]

    def save_job(self, job):
        self._writable()
        if job['state'] not in {'ready','waiting_example','deferred','done'} or job['kind'] not in {'build','repair','trial'}:
            raise ValueError('Unknown realization job state/kind')
        self.db.execute('INSERT OR REPLACE INTO realization_jobs VALUES(?,?)', (job['id'], json.dumps(job, ensure_ascii=False)))
        self.db.commit()
        return job

    def save_case(self, task, experience):
        from dataclasses import asdict
        self._writable()
        if task.split != 'train': raise ValueError('Only completed Train cases are eligible')
        self.db.execute('INSERT OR IGNORE INTO train_cases VALUES(?,?)',
                        (task.physical_key, json.dumps({'task': asdict(task), 'experience': experience}, ensure_ascii=False)))
        self.db.commit()

    def train_cases(self):
        if not self.db.execute("SELECT 1 FROM sqlite_master WHERE name='train_cases'").fetchone(): return []
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM train_cases ORDER BY rowid')]

    def digest(self):
        rows = {table: list(self.db.execute(f"SELECT * FROM {table} ORDER BY rowid"))
                for table in ('metadata', 'assets', 'attempts')}
        for table in ['realization_jobs', 'train_cases']:
            if self.db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (table,)).fetchone():
                content = list(self.db.execute(f'SELECT * FROM {table} ORDER BY rowid'))
                if content: rows[table] = content
        return digest(rows)

    def freeze(self, destination):
        self._writable()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        target = sqlite3.connect(destination / "bank.sqlite3")
        self.db.backup(target)
        target.execute('DROP TABLE IF EXISTS realization_jobs')
        target.execute('DROP TABLE IF EXISTS train_cases')
        usable = {p['id'] for p in self.all('program') if p['state'] == 'usable'}
        for kind, asset_id, payload in target.execute('SELECT kind,id,payload FROM assets').fetchall():
            asset = json.loads(payload)
            if (kind == 'program' and asset_id not in usable) or (
                    kind == 'implementation' and asset['program_id'] not in usable):
                target.execute('DELETE FROM assets WHERE id=?', (asset_id,))
            elif kind == 'workflow':
                changed = False
                for node in asset['nodes']:
                    if node.get('program_id') and node['program_id'] not in usable:
                        interface = resolve_node_interface(node, self)
                        node.pop('program_id')
                        node.pop('skill_id', None)
                        node.pop('dynamic', None)
                        node.update(execution_mode='dynamic', goal=interface['node_goal'])
                        if interface['bound_skill_id']:
                            node['reference_skill_ids'] = list(dict.fromkeys([
                                *node.get('reference_skill_ids', []), interface['bound_skill_id']]))[:3]
                        changed = True
                if changed:
                    asset.pop('id')
                    asset['source_workflow_id'] = asset_id
                    asset['id'] = 'workflow_' + digest(asset)
                    target.execute('DELETE FROM assets WHERE id=?', (asset_id,))
                    target.execute('INSERT INTO assets VALUES(?,?,?)', ('workflow', asset['id'], json.dumps(asset)))
        for program_id, in target.execute('SELECT DISTINCT program_id FROM attempts').fetchall():
            if program_id not in usable:
                target.execute('DELETE FROM attempts WHERE program_id=?', (program_id,))
        target.commit()
        seed = int(self.db.execute("SELECT value FROM metadata WHERE key='seed'").fetchone()[0])
        frozen = Bank(destination, readonly=True, seed=seed)
        manifest = {"schema": "empirical.bank.v1", "digest": frozen.digest(), 'source_digest': self.digest(),
                    "programs": [{"id": p["id"], "state": p["state"]} for p in frozen.all("program")]}
        frozen.close()
        target.close()
        (destination / "freeze.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def close(self):
        self.db.close()
