"""One version state and one attempt table; no per-layer qualification."""
import json
import sqlite3
from pathlib import Path

from .contracts import digest, validate_program, validate_workflow


class Bank:
    def __init__(self, root, *, readonly=False, seed=42):
        self.root, self.readonly = Path(root), readonly
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
            asset["id"] = "program_" + validate_program(asset)
            asset["state"] = "candidate"
        elif kind == "workflow":
            validate_workflow(asset, [p["id"] for p in self.all("program")], [s["id"] for s in self.all("skill")])
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
        return asset

    def get(self, asset_id):
        row = self.db.execute("SELECT payload FROM assets WHERE id=?", (asset_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def all(self, kind):
        return [json.loads(r[0]) for r in self.db.execute("SELECT payload FROM assets WHERE kind=? ORDER BY id", (kind,))]

    def routes(self, node, *, allow_candidate=False):
        refs = [node["program_id"]] if node.get("program_id") else [
            i["program_id"] for i in self.all("implementation") if i["skill_id"] == node.get("skill_id")]
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

    def retrieve(self, query, limit=8):
        words = set(query.lower().split())
        assets = [*self.all("skill"), *self.all("workflow")]
        return sorted(assets, key=lambda a: (-len(words & set((a.get("goal", "") + " " +
                     a.get("guidance", "")).lower().split())), a["id"]))[:limit]

    def digest(self):
        return digest({table: list(self.db.execute(f"SELECT * FROM {table} ORDER BY rowid"))
                       for table in ("metadata", "assets", "attempts")})

    def freeze(self, destination):
        self._writable()
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=False)
        target = sqlite3.connect(destination / "bank.sqlite3")
        self.db.backup(target)
        target.close()
        manifest = {"schema": "empirical.bank.v1", "digest": self.digest(),
                    "programs": [{"id": p["id"], "state": p["state"]} for p in self.all("program")]}
        (destination / "freeze.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def close(self):
        self.db.close()
