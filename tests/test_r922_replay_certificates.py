"""Linear admission, immutable executable identity and actual Direct execution."""
import copy
from dataclasses import replace
from types import SimpleNamespace

from atomic_skillgraph.core.bindings import BindingExpression, BindingExprKind
from atomic_skillgraph.core.results import RuntimeOccurrence, RuntimeLinearPlan
from atomic_skillgraph.core.status import ToolStatus
from atomic_skillgraph.evolution.aligner import _tool_signature
from atomic_skillgraph.evolution.replay import ReplayCaseResult, replay_case_id
from atomic_skillgraph.evolution.replay_certificates import ReplayCertificates
from atomic_skillgraph.evolution.tool_compiler import ToolCompiler
from atomic_skillgraph.runtime.budget import RuntimeBudget
from atomic_skillgraph.runtime.task_context import TaskRuntimeContext
from atomic_skillgraph.system import AtomicSkillGraphSystem
from experiments.fakes import FakeHarness, fake_task
from experiments.report import summarize_traces
from test_batch_evolution import _system_config
from test_r92_replay_source_authority import _source


def test_original_discovery_reprocessing_is_idempotent(tmp_path):
    config = _system_config(tmp_path / "bank")
    with AtomicSkillGraphSystem(config, harness=FakeHarness()) as system:
        task = fake_task("original", "apple_1")
        compiled = ToolCompiler().compile([_source("original", "source_original", "object", "apple_1")])[0]
        system._replay_source_authority = lambda: SimpleNamespace(resolve=lambda *args, **kwargs: task)
        physical = []
        def execute(task, tool, case, *, requested_task_id):
            physical.append(case["case_id"])
            return ReplayCaseResult(case["case_id"], case["trace_id"], task.task_id,
                task.task_id, task.task_id, "final_validation", True)
        system._replay_tool_candidate_result = execute
        prepared = SimpleNamespace(compiled=[compiled], composite=None, source_composite_ref="")
        counts = []
        for index in range(2):
            trace = system.orchestrator.create_trace_builder(task).trace
            trace.trace_id = "source_original"
            system._apply_evolution(prepared, trace, task)
            if index == 0:
                trace.finish()
                system.traces.save_atomic(trace)
                system._commit_replay_certificates(trace)
            counts.append(system.ledger.count())
        assert counts[0] == counts[1]
        assert len(physical) == len(system.tools.list_refs()) == 1


def test_ten_cases_linear_certificates_resume_changed_program_and_direct(tmp_path):
    config = _system_config(tmp_path / "bank")
    physical = []
    traces = []
    with AtomicSkillGraphSystem(config, harness=FakeHarness()) as system:
        # Source resolution and real replay execution have their own integration
        # tests; count calls at that physical boundary, not at admission entry.
        system._replay_source_authority = lambda: SimpleNamespace(resolve=lambda case, **kwargs: fake_task(case["source_task"]["task_id"], "apple_1"))
        def execute(task, tool, case, *, requested_task_id):
            physical.append(replay_case_id(case))
            return ReplayCaseResult(replay_case_id(case), case["trace_id"], task.task_id,
                requested_task_id, task.task_id, "final_validation", True, started=True,
                executed_action_count=1, completed=True, atomic_effect_passed=True, output_validation_passed=True)
        system._replay_tool_candidate_result = execute
        original = ToolCompiler().compile([_source("case0", "source0", "object", "apple_1")])[0]
        atomic_ref = system.aligner.align_atomic(original.atomic)
        refs, impls, cases = set(), set(), []

        def admit(case, executable=original.tool):
            task = fake_task(case["source_task"]["task_id"], "apple_1")
            trace = system.orchestrator.create_trace_builder(task).trace
            candidate = replace(executable, tests=[case], status=ToolStatus.ADMISSION_PENDING)
            admitted = system.admission.admit_tool(candidate, replay=lambda tool, item:
                system._replay_case_with_source_authority(tool, item, current_task=task,
                    current_trace=None, audit_trace=trace))
            assert admitted.status is ToolStatus.CANDIDATE
            alignment = system.aligner.align_tool_with_replays(admitted, admission=system.admission, replay=None)
            system._capture_replay_bank_metrics(trace)
            trace.finish()
            system.traces.save_atomic(trace)
            system._commit_replay_certificates(trace)
            traces.append(trace)
            return alignment.ref, trace

        for number in range(10):
            occurrence = _source(f"case{number}", f"source{number}", "object", f"apple_{number+1}")
            case = ToolCompiler().compile([occurrence])[0].tool.tests[0]
            cases.append(case)
            ref, _ = admit(case)
            refs.add(ref)
            implementation = system.admission.admit_implementation(original.implementation,
                replace(system.tools.get(ref), ref=original.tool.ref), atomic=original.atomic,
                harness=system.harness)
            assert implementation.status.value == "candidate", implementation
            impls.add(system.aligner.align_implementation(implementation, atomic_ref, ref))
        assert len(physical) == 10 and len(refs) == len(impls) == 1
        certificates = ReplayCertificates(system.ledger)
        signature = _tool_signature(original.tool)
        assert len(certificates.events(signature)) == 10
        assert len(system.tools.tools_with_replay_evidence()[0].tests) == 10
        assert len(system.tools.get(next(iter(refs))).tests) == 1
        _, duplicate = admit(cases[4])
        assert len(physical) == 10
        assert duplicate.metadata["replay_accounting"]["replay_certificate_reuses"] == 1
        assert duplicate.metadata.get("tool_replay_results", []) == []
        # A new store instance (resume) sees the same certificates; publishing
        # the same trace's certificates twice is an append-only no-op.
        system._commit_replay_certificates(traces[0])
        assert len(ReplayCertificates(system.ledger).events(signature)) == 10
        assert ReplayCertificates(system.ledger).lookup(signature, cases[4]) is not None
        assert ReplayCertificates(system.ledger, authority_version="changed").lookup(signature, cases[4]) is None
        tampered = copy.deepcopy(cases[4])
        tampered["bindings"]["object"] = "unobserved_999"
        assert certificates.lookup(signature, tampered) is None

        # Two or more independent cases still expose one actual invocation.
        task = fake_task("direct", "apple_99")
        atomic = system.skills.get_atomic(atomic_ref)
        occurrence = RuntimeOccurrence("node", "node", atomic_ref, [],
            {"object": BindingExpression(BindingExprKind.CONSTANT, constant="apple_99")},
            list(impls), atomic.effects)
        plan = RuntimeLinearPlan(task.task_id, "stored_composite", "", [occurrence], ["node"], [], [], system.harness.task_contract(task), {})
        ctx = TaskRuntimeContext.create(task, plan, system.harness,
            system.orchestrator.create_trace_builder(task), RuntimeBudget(global_action_budget=100, node_action_budget=35))
        ctx.budget.begin_node("node")
        ctx.binding_store.resolve_occurrence_specs(occurrence, ctx.world_revision)
        ctx.begin_occurrence(occurrence)
        invocations = system.invocation_compiler.compile_candidates(occurrence, ctx.binding_store, task_id=task.task_id)
        assert len(invocations) == 1
        direct = system.orchestrator.node_executor.try_autonomous(occurrence, invocations, ctx)
        assert direct is not None and direct.atomic_effect_passed, direct
        assert direct.node_status.value == "direct_autonomous_success"
        assert not ctx.trace_builder.trace.agent_sessions

        changed = copy.deepcopy(original.tool)
        changed.artifact["steps"] = [*changed.artifact["steps"], *copy.deepcopy(changed.artifact["steps"])]
        assert _tool_signature(changed) != signature
        for case in cases:
            ref, _ = admit(case, changed)
            assert ref not in refs
        assert len(physical) == 20
        assert len(certificates.events(_tool_signature(changed))) == 10
        summary = summarize_traces(traces)
        assert summary["fresh_replay_executions"] == 20
        assert summary["replay_certificate_reuses"] == 1
        assert summary["unique_executable_tools"] == 2
        assert summary["replay_evidence_count"] == 20

        before = system.knowledge_digest()
        system.readonly = True
        new_case = copy.deepcopy(cases[0])
        new_case["case_id"] = "frozen_new_case"
        audit = system.orchestrator.create_trace_builder(task).trace
        result = system._replay_case_with_source_authority(original.tool, new_case,
            current_task=task, current_trace=None, audit_trace=audit)
        assert result.passed
        system._commit_replay_certificates(audit)
        assert before == system.knowledge_digest()
        assert "replay_certificate_events" not in audit.metadata
