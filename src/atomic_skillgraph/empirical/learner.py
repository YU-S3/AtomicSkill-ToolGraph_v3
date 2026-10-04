"""One experience proposal, one Builder and at most two fixed Train trials."""
from copy import deepcopy

from .contracts import TrialCase, digest
from .prompts import BUILD, BUILDER_PROMPT, LEARNING, LEARNER_PROMPT


class Learner:
    def __init__(self, system):
        self.system = system
        self.cases = []

    def learn(self, task, trace, *, focus=None):
        s, bank = self.system, self.system.bank
        public_case = {"task": {"goal": task.goal, "inputs": task.inputs},
                       "events": trace["tools"], "score": trace["score"], "result": trace["execution"]}
        if not any(t.physical_key == task.physical_key for t, _ in self.cases):
            self.cases.append((task, public_case))
        related = bank.retrieve(task.goal)
        tools = getattr(s.adapter, 'tool_definitions', s.adapter.available_tools)()
        log = {"decision": None, "program": None, "tests": [], "errors": []}
        def receive(key, stage, prompt, material, name, schema, repair_limit):
            checkpoint = s.checkpoint
            if checkpoint and key in checkpoint.state:
                return checkpoint.state[key]
            value = s.agent(stage, prompt, material, name, schema, repair_limit=repair_limit)
            if checkpoint:
                checkpoint.advance(checkpoint.state['stage'], **{key: value})
            return value
        proposal = receive('learning_proposal', "extractor", LEARNER_PROMPT, {"experience": public_case,
            "related": related, "tools": tools, "selected_local_goal": focus}, "submit_learning", LEARNING,
            1)
        log["decision"] = proposal["decision"]
        if proposal["decision"] == "no_change":
            return log
        skill = None
        if proposal.get("skill"):
            skill = bank.put("skill", proposal["skill"])
        elif proposal.get("existing_skill_id"):
            skill = bank.get(proposal["existing_skill_id"])
            if skill is None:
                raise ValueError("Learner selected an unknown Skill")
        if proposal["decision"] == "propose_skill_and_program_spec" and proposal.get("generate_program", True):
            if skill is None:
                raise ValueError("New program needs a Skill interface")
            # Test subjects are fixed before the Builder or any tests. Source
            # retries and same physical tasks never create additional credit.
            subjects, keys = [], set()
            words = set((skill["goal"] + " " + task.goal).casefold().split())
            related_cases = sorted(enumerate(self.cases), key=lambda row: (
                -len(words & set(row[1][0].goal.casefold().split())), -row[0]))
            for _, case in related_cases:
                if case[0].physical_key not in keys:
                    subjects.append(case)
                    keys.add(case[0].physical_key)
                if len(subjects) == 2:
                    break
            log["test_subjects"] = [case[0].physical_key for case in subjects]
            base = {"entry": "run", "backend": "sandbox_python_v1", "input_schema": skill["input_schema"],
                    "output_schema": skill["output_schema"], "allowed_tools": [t["name"] for t in tools if t['name'] != 'execute_python'],
                    "environment": s.config["program_environment"]}
            repaired = False
            material = {"skill": skill, "tools": tools, "examples": [
                {'case_id': case[0].physical_key, **case[1]} for case in subjects]}
            for generation in range(2):
                generated = receive('builder_' + str(generation), "tool_builder", BUILDER_PROMPT,
                                    material, "submit_program", BUILD, 0)
                bindings = {row['case_id']: row for row in generated['trial_inputs']}
                if len(bindings) != len(generated['trial_inputs']) or set(bindings) != keys:
                    raise ValueError('Builder must bind exactly the fixed TrialCases')
                log['trial_inputs'] = generated['trial_inputs']
                log['example_inputs'] = bindings[subjects[0][0].physical_key]['inputs']
                candidate = {**base, "source": generated["source"]}
                try:
                    program = bank.put("program", candidate)
                except (ValueError, SyntaxError) as exc:
                    log["errors"].append(str(exc))
                    if repaired:
                        break
                    repaired = True
                    material = {"skill": skill, "tools": tools, 'examples': material.get('examples', []),
                                "source": candidate["source"], "error": str(exc)}
                    continue
                bank.put("implementation", {"skill_id": skill["id"], "program_id": program["id"]})
                log["program"] = program["id"]
                failures = []
                for index, (test_task, _) in enumerate(subjects):
                    if any(row['task_key'] == test_task.physical_key and row['origin'] == 'train_test'
                           for row in bank.attempts(program['id'])):
                        continue
                    binding = bindings[test_task.physical_key]
                    trial_case = TrialCase(test_task.physical_key, test_task.physical_key, test_task.task_id,
                        deepcopy(binding['inputs']), binding['start_mode'], tuple(binding['prefix']))
                    if trial_case.prefix and list(trial_case.prefix) != [
                            {'name': e['name'], 'arguments': e['arguments']} for e in subjects[index][1]['events'][:len(trial_case.prefix)]]:
                        raise ValueError('Trial prefix must match this case real public actions')
                    trial = s.test_program(program, trial_case.inputs, test_task,
                        trial_id=digest([program['id'], test_task.physical_key, 'train_test']), trial_case=trial_case)
                    log["tests"].append(trial)
                    if trial["outcome"] == "execution_failure":
                        failures.append(trial)
                if failures and not repaired:
                    repaired = True
                    material = {"skill": skill, "tools": tools, "source": candidate["source"],
                                'examples': [{'case_id': case[0].physical_key, **case[1]} for case in subjects],
                                "trial_inputs": generated["trial_inputs"], "errors": failures}
                    continue
                break
        if proposal.get("workflow"):
            workflow = deepcopy(proposal["workflow"])
            for node in workflow["nodes"]:
                if node.get("skill_id") == "$new":
                    if skill is None:
                        raise ValueError("Workflow references a missing proposed Skill")
                    node["skill_id"] = skill["id"]
            try:
                log["workflow"] = bank.put("workflow", workflow)["id"]
            except ValueError as exc:
                log['errors'].append('Workflow rejected: ' + str(exc))
        return log
