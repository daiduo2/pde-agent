"""Curated scientific tool catalog for Claude SDK and Hypothesis Lab agents."""


import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STATE_HINT = "SCIENTIFIC_TOOL_CATALOG_BUILT"
VALID_RISK_LEVELS = {
    "read_only",
    "artifact_write",
    "sandbox",
    "long_running",
    "external_network",
}


class ScientificToolRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScientificToolSpec:
    name: str
    command: str
    description: str
    risk_level: str
    required_inputs: list[str]
    produced_artifacts: list[str]
    state_hints: list[str]
    agent_guidance: str
    runtime_status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ScientificToolRouterError(
                f"unsupported risk_level for {self.name}: {self.risk_level}"
            )
        return asdict(self)


def default_scientific_tool_specs(
    *,
    include_compatibility_tools: bool = False,
) -> list[ScientificToolSpec]:
    specs = [
        ScientificToolSpec(
            name="summarize-hypothesis-lab-trace",
            command="summarize-hypothesis-lab-trace",
            description="Summarize append-only Hypothesis Lab trace events.",
            risk_level="read_only",
            required_inputs=["trace_path"],
            produced_artifacts=["trace_summary_json"],
            state_hints=["HYPOTHESIS_LAB_TRACE_SUMMARIZED"],
            agent_guidance=(
                "Use before planning another iteration so repeated failures and "
                "promoted candidates are visible in context."
            ),
        ),
        ScientificToolSpec(
            name="detect-hypothesis-doom-loop",
            command="detect-hypothesis-doom-loop",
            description="Detect repeated non-improving scientific candidate families.",
            risk_level="read_only",
            required_inputs=["trace_path"],
            produced_artifacts=["doom_loop_report_json"],
            state_hints=["HYPOTHESIS_DOOM_LOOP_CHECKED"],
            agent_guidance=(
                "If guard_active is true, obey blocked_candidate_families and "
                "switch to a different mechanism family."
            ),
        ),
        ScientificToolSpec(
            name="build-hypothesis-kpi-report",
            command="build-hypothesis-kpi-report",
            description="Build a quantitative KPI report from Hypothesis Lab trace events.",
            risk_level="read_only",
            required_inputs=["trace_path"],
            produced_artifacts=[
                "hypothesis_kpi_report_json",
                "active_audit_repair_errors",
                "repair_nonconvergence_errors",
                "repair_convergence_errors",
                "audit_repair_error_counts",
                "audit_repair_consumed_error_counts",
                "audit_repair_converged_error_counts",
                "persistent_search_guidance",
                "search_guidance_action_counts",
                "cluster_saturation_signals",
                "saturated_candidate_families",
            ],
            state_hints=["HYPOTHESIS_KPI_REPORT_BUILT"],
            agent_guidance=(
                "Use after each batch to decide whether the research loop is "
                "improving, regressing, or over-exploring failed families. Inspect "
                "active_audit_repair_errors, repair_nonconvergence_errors, and "
                "repair_convergence_errors before planning candidates so repair "
                "state is not lost between audit and the next Claude SDK agent "
                "iteration. Preserve "
                "persistent_search_guidance, "
                "cluster_saturation_signals, and saturated_candidate_families when "
                "calling plan-hypothesis-candidates."
            ),
        ),
        ScientificToolSpec(
            name="evaluate-task1-agent-capability",
            command="evaluate-task1-agent-capability",
            description=(
                "Evaluate whether an audited Task1 agent loop improved score "
                "through reusable Claude SDK agent capability rather than "
                "score-only tuning."
            ),
            risk_level="read_only",
            required_inputs=[
                "chain_report_path",
                "trace_path",
                "output_path",
            ],
            produced_artifacts=[
                "task1_agent_capability_benchmark_json",
                "score_progress",
                "candidate_generation",
                "sandbox_validation",
                "learning_memory",
                "promotion_quality",
                "capability_checks",
            ],
            state_hints=["TASK1_AGENT_CAPABILITY_BENCHMARKED"],
            agent_guidance=(
                "Run after each audited Task1 optimization chain. Treat "
                "score_only as a warning that Task1 improved without enough "
                "evidence-backed candidate diversity. Treat infrastructure_only "
                "as a signal to expand candidate families or validation slices. "
                "Use passed as the gate before starting a longer autonomous "
                "Task1 capability goal. This evaluates the current Claude SDK "
                "agent substrate."
            ),
        ),
        ScientificToolSpec(
            name="generate-task1-candidate-families",
            command="generate-task1-candidate-families",
            description=(
                "Generate bounded Task1 CandidateFamilyCard artifacts from a "
                "Task1 capability benchmark and runtime audit-chain report."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "benchmark_path",
                "chain_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "candidate_family_report_json",
                "candidate_families",
                "generation_context",
            ],
            state_hints=["TASK1_CANDIDATE_FAMILIES_GENERATED"],
            agent_guidance=(
                "Use when Task1 capability benchmark returns infrastructure_only "
                "or stagnated. Treat generated cards as bounded candidate-family "
                "hypotheses that must be sandboxed before promotion."
            ),
        ),
        ScientificToolSpec(
            name="run-candidate-family-sandbox",
            command="run-candidate-family-sandbox",
            description=(
                "Expand Task1 CandidateFamilyCard artifacts into bounded sandbox "
                "plans without executing arbitrary generated code."
            ),
            risk_level="sandbox",
            required_inputs=["family_report_path", "output_path"],
            produced_artifacts=[
                "candidate_family_sandbox_json",
                "sandbox_items",
                "planned_count",
                "blocked_count",
            ],
            state_hints=["TASK1_CANDIDATE_FAMILY_SANDBOX_PLANNED"],
            agent_guidance=(
                "Run immediately after candidate-family generation. Planned "
                "items can be fed into bounded Claude SDK agent iterations; blocked "
                "items should be revised or routed to executor authoring."
            ),
        ),
        ScientificToolSpec(
            name="materialize-candidate-family-item",
            command="materialize-candidate-family-item",
            description=(
                "Convert one planned Task1 candidate-family sandbox item into "
                "an ALGORITHM_CANDIDATES_COMPILED artifact consumable by "
                "run-hypothesis-lab."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "sandbox_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "algorithm_candidates_json",
                "selected_candidate_family_id",
                "agent_runtime_handoff",
            ],
            state_hints=["ALGORITHM_CANDIDATES_COMPILED"],
            agent_guidance=(
                "Use after run-candidate-family-sandbox and before "
                "run-hypothesis-lab. This is the bounded bridge from generated "
                "candidate-family plans into executable Hypothesis Lab candidates."
            ),
        ),
        ScientificToolSpec(
            name="build-hypothesis-lab-next-policy",
            command="build-hypothesis-lab-next-policy",
            description=(
                "Convert Hypothesis Lab failure memory or run reports into a "
                "next-step policy for the Claude SDK agent loop."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_path",
                "failure_memory_path_or_hypothesis_lab_report_path",
            ],
            produced_artifacts=[
                "hypothesis_lab_next_policy_json",
                "next_action",
                "next_command",
                "learning_objective",
            ],
            state_hints=["HYPOTHESIS_LAB_NEXT_POLICY_BUILT"],
            agent_guidance=(
                "Run after each Hypothesis Lab sandbox result so negative and "
                "positive evidence changes the next agent policy instead of "
                "being left for a human supervisor to interpret."
            ),
        ),
        ScientificToolSpec(
            name="record-agent-runtime-issue",
            command="record-agent-runtime-issue",
            description=(
                "Record a structured Claude SDK runtime issue through the "
                "controlled pde-agent CLI instead of letting the agent write "
                "free-form files."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_path",
                "failed_command",
                "stderr_summary",
                "next_action",
            ],
            produced_artifacts=[
                "runtime_issue_json",
                "failed_command",
                "likely_schema_gap",
                "next_action",
            ],
            state_hints=["AGENT_RUNTIME_ISSUE_RECORDED"],
            agent_guidance=(
                "Use exactly once when an allowed closure-loop command fails. "
                "After recording the issue, stop instead of probing help commands "
                "or trying alternate shell diagnostics."
            ),
        ),
        ScientificToolSpec(
            name="evaluate-candidate-factory-capability",
            command="evaluate-candidate-factory-capability",
            description=(
                "Evaluate whether generated Task1 candidate families added "
                "diverse executable search surface for the Claude SDK agent."
            ),
            risk_level="read_only",
            required_inputs=[
                "family_report_path",
                "sandbox_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "candidate_factory_capability_json",
                "candidate_space",
                "sandbox_readiness",
                "capability_checks",
            ],
            state_hints=["TASK1_CANDIDATE_FACTORY_CAPABILITY_EVALUATED"],
            agent_guidance=(
                "Use before restarting agent iteration after candidate-space exhaustion. "
                "A passed report means the next long run has new bounded candidate "
                "surface; a failed report means generate or revise families first."
            ),
        ),
        ScientificToolSpec(
            name="build-agent-research-directive",
            command="build-agent-research-directive",
            description=(
                "Build a higher-level Claude SDK agent research directive from "
                "Task1 capability and candidate-factory evidence."
            ),
            risk_level="read_only",
            required_inputs=[
                "task1_benchmark_path",
                "candidate_factory_benchmark_path",
                "output_path",
            ],
            produced_artifacts=[
                "research_directive_json",
                "research_focus",
                "candidate_generator",
                "learning_objective",
                "hypothesis_candidate_options",
            ],
            state_hints=["AGENT_RESEARCH_DIRECTIVE_BUILT"],
            agent_guidance=(
                "Use after capability or candidate-factory evaluation. The "
                "directive should guide the next Claude SDK agent run at the "
                "research-plan level, not bypass sandbox or promotion gates."
            ),
        ),
        ScientificToolSpec(
            name="evaluate-research-directive-followthrough",
            command="evaluate-research-directive-followthrough",
            description=(
                "Evaluate whether a runtime audit-chain followed a legacy "
                "research directive and preserved audit cleanliness."
            ),
            risk_level="read_only",
            required_inputs=[
                "directive_path",
                "chain_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "research_directive_followthrough_json",
                "candidate_generator_followed",
                "candidate_options_forwarded",
                "audit_clean",
            ],
            state_hints=[
                "AGENT_RESEARCH_DIRECTIVE_FOLLOWTHROUGH_EVALUATED"
            ],
            agent_guidance=(
                "Run after a directive-driven compatibility chain. If followthrough "
                "fails, revise directive routing before running more parameter "
                "search."
            ),
        ),
        ScientificToolSpec(
            name="build-agent-meta-research-directive",
            command="build-agent-meta-research-directive",
            description=(
                "Route the Claude SDK agent from experiment execution toward research "
                "organization by turning followthrough, chain, Task1, and "
                "candidate-factory evidence into the next bounded directive."
            ),
            risk_level="read_only",
            required_inputs=[
                "followthrough_report_path",
                "chain_report_path",
                "task1_benchmark_path",
                "candidate_factory_benchmark_path",
                "output_path",
            ],
            produced_artifacts=[
                "meta_research_directive_json",
                "research_assessment",
                "research_focus",
                "candidate_generator",
            ],
            state_hints=["AGENT_META_RESEARCH_DIRECTIVE_BUILT"],
            agent_guidance=(
                "Use after evaluating directive followthrough. It chooses "
                "whether the Claude SDK agent should continue, resolve guard "
                "conflict, or expand candidate space before launching the next "
                "runtime chain."
            ),
        ),
        ScientificToolSpec(
            name="build-repair-policy-summary",
            command="build-repair-policy-summary",
            description=(
                "Summarize converged and nonconverged audit repairs into "
                "a compact policy artifact for the next planning round."
            ),
            risk_level="read_only",
            required_inputs=["trace_path"],
            produced_artifacts=[
                "repair_policy_summary_json",
                "converged_repair_errors",
                "nonconverged_repair_errors",
                "repair_error_stats",
                "recommended_actions",
            ],
            state_hints=["AGENT_REPAIR_POLICY_SUMMARY_BUILT"],
            agent_guidance=(
                "Run after runtime audit-chain or auto-repair events and before "
                "candidate planning. Use recommended_actions to avoid repeating "
                "nonconverged repairs, and keep converged repair paths as positive "
                "memory for the next Claude Agent SDK runtime pass."
            ),
        ),
        ScientificToolSpec(
            name="plan-hypothesis-candidates",
            command="plan-hypothesis-candidates",
            description=(
                "Convert doom-loop guard, KPI memory, and candidate registry "
                "metadata into an auditable candidate-family schedule."
            ),
            risk_level="read_only",
            required_inputs=["candidate_generator", "doom_loop_report_path"],
            produced_artifacts=[
                "candidate_plan_json",
                "candidate_generator_schedule",
                "hypothesis_lab_policy_constraints",
                "audit_repair_tasks",
                "audit_repair_policies",
                "repair_policy_recommended_actions",
                "repair_nonconvergence_errors",
                "repair_convergence_errors",
                "audit_repair_converged_error_counts",
                "research_chronicle_repair_policy_memory",
            ],
            state_hints=["HYPOTHESIS_CANDIDATE_PLAN_BUILT"],
            agent_guidance=(
                "Use between KPI/doom-loop checks and Claude SDK execution. If "
                "expansion_required is true, route to strategy or literature "
                "expansion instead of repeatedly sampling depleted local families. "
                "Inspect candidate_generator_schedule, audit_repair_policies, and "
                "repair_nonconvergence_errors directly from the command payload "
                "before launching the next SDK run. If a repair_policy_summary_path is "
                "available, pass it in so recommended_actions and converged repair "
                "memory are preserved. Preserve repair_convergence_errors so proven "
                "audit-repair paths remain available as positive memory. If a prior "
                "research_chronicle_path exists, pass it so the Repair Policy Memory "
                "section can constrain the next candidate plan. After "
                "build-hypothesis-lab-next-policy, pass "
                "--hypothesis-lab-next-policy-path so the next candidate plan "
                "uses the sandbox evidence rather than restarting from defaults."
            ),
        ),
        ScientificToolSpec(
            name="build-fno-training-harness-contract",
            command="build-fno-training-harness-contract",
            description=(
                "Build a Claude SDK agent-facing FNO training action contract with explicit "
                "data paths, budget parameters, output paths, and optional next "
                "actions. This command does not run training."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_path",
                "workspace_dir",
                "source_path",
                "previous_evidence_paths",
                "budget_parameters",
            ],
            produced_artifacts=[
                "harness_contract_path",
                "available_actions",
                "budget_contract",
                "data_contract",
            ],
            state_hints=["FNO_TRAINING_HARNESS_CONTRACT_BUILT"],
            agent_guidance=(
                "Use before FNO training optimization when the agent needs to choose "
                "actions itself. The contract is not a fixed pipeline: inspect "
                "available_actions, execute only the selected action command, then "
                "choose the next action from next_action_options. Do not create "
                "directories with shell commands and do not probe CLI help."
            ),
        ),
        ScientificToolSpec(
            name="run-fno-training-optimization-loop",
            command="run-fno-training-optimization-loop",
            description=(
                "Run bounded FNO short-window training candidates from an explicit "
                "candidate plan and record candidate checkpoints, logs, and scores."
            ),
            risk_level="sandbox",
            required_inputs=[
                "candidate_plan_path",
                "source_path",
                "output_dir",
                "budget_contract",
            ],
            produced_artifacts=[
                "optimization_report_path",
                "candidate_reports",
                "training_logs",
                "checkpoints",
                "short_window_predictions",
            ],
            state_hints=["FNO_TRAINING_OPTIMIZATION_LOOP_RAN"],
            agent_guidance=(
                "Use only after build-fno-training-harness-contract or "
                "plan-fno-training-candidates has produced a candidate_plan_path. "
                "Keep data limits, train steps, and batch size within the prior "
                "budget_contract; do not invent new data paths, create directories "
                "manually, or run unbounded training."
            ),
        ),
        ScientificToolSpec(
            name="record-fno-training-optimization-evidence",
            command="record-fno-training-optimization-evidence",
            description=(
                "Convert a bounded FNO training optimization report into structured "
                "evidence records for Hypothesis Lab algorithm design and promotion."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "optimization_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "fno_training_optimization_evidence_json",
                "candidate_records",
                "training_directive",
            ],
            state_hints=["FNO_TRAINING_OPTIMIZATION_EVIDENCE_RECORDED"],
            agent_guidance=(
                "Use after run-fno-training-optimization-loop. Pass the generated "
                "optimization report and then feed this evidence to "
                "build-hypothesis-algorithm-design with --evidence-path so model "
                "training results become part of the hypothesis and promotion loop."
            ),
        ),
        ScientificToolSpec(
            name="audit-agent-runtime-availability",
            command="audit-agent-runtime-availability",
            description=(
                "Build a runtime availability matrix for Claude Agent SDK or a "
                "compatibility backend from CLI exposure, tool catalog exposure, "
                "required commands, and canary reports."
            ),
            risk_level="read_only",
            required_inputs=[
                "runtime_name",
                "tool_catalog_path",
                "canary_report_paths",
                "cli_commands",
                "required_commands",
            ],
            produced_artifacts=[
                "availability_audit_path",
                "runtime_metadata",
                "availability_matrix",
                "missing_required_commands",
                "canary_evidence",
            ],
            state_hints=["AGENT_RUNTIME_AVAILABILITY_AUDITED"],
            agent_guidance=(
                "Use this as the primary availability gate for claude_agent_sdk. "
                "A command being CLI-exposed is not enough; required commands "
                "should also be tool-catalog visible and supported by passed "
                "runtime canary evidence."
            ),
        ),
        ScientificToolSpec(
            name="build-agent-runtime-contract",
            command="build-agent-runtime-contract",
            description=(
                "Build a runtime contract for a primary Claude Agent SDK backend "
                "using the pde-agent tool catalog as the execution surface."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "runtime_name",
                "workspace_dir",
                "tool_catalog_path",
                "required_commands",
                "output_path",
            ],
            produced_artifacts=[
                "runtime_contract_path",
                "allowed_tools",
                "allowed_bash_prefixes",
                "required_commands",
            ],
            state_hints=["AGENT_RUNTIME_CONTRACT_BUILT"],
            agent_guidance=(
                "Use this before starting a claude_agent_sdk run. The contract "
                "makes Claude Agent SDK the active agent substrate. Legacy "
                "runtime commands are compatibility only and must not be used "
                "for current Task1 training iterations. Keep Bash restricted to "
                "uv run --frozen pde-agent commands."
            ),
        ),
        ScientificToolSpec(
            name="run-claude-agent-sdk-runtime",
            command="run-claude-agent-sdk-runtime",
            description=(
                "Run or dry-run the Claude Agent SDK backend against a bounded "
                "prompt, tool policy, and pde-agent Bash prefix policy."
            ),
            risk_level="sandbox",
            required_inputs=[
                "prompt",
                "workspace_dir",
                "allowed_tools",
                "disallowed_tools",
                "allowed_bash_prefixes",
                "output_path",
            ],
            produced_artifacts=[
                "claude_agent_sdk_session_path",
                "sdk_messages",
                "tool_policy",
                "bash_policy",
            ],
            state_hints=["CLAUDE_AGENT_SDK_RUNTIME_RAN"],
            agent_guidance=(
                "Use this to let Claude Agent SDK directly drive pde-agent tools. "
                "For migration smokes, keep --dry-run until the runtime contract "
                "and availability audit pass. Bash should remain limited to "
                "uv run --frozen pde-agent prefixes."
            ),
        ),
        ScientificToolSpec(
            name="build-claude-fno-training-smoke-task",
            command="build-claude-fno-training-smoke-task",
            description=(
                "Build a prompt and expected-artifact checklist for a Claude "
                "Agent SDK FNO training smoke that drives training, evidence, "
                "Hypothesis Lab design, validation, and promotion/evidence gates."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_dir",
                "workspace_dir",
                "runtime_contract_path",
                "tool_catalog_path",
                "source_path",
            ],
            produced_artifacts=[
                "claude_agent_prompt_md",
                "expected_artifacts_json",
                "required_command_flow",
            ],
            state_hints=["CLAUDE_FNO_TRAINING_SMOKE_TASK_BUILT"],
            agent_guidance=(
                "Use this to prepare the first real Claude Agent SDK canary for "
                "the FNO training loop. The generated prompt is for the SDK "
                "runtime, not for Codex to execute manually."
            ),
        ),
        ScientificToolSpec(
            name="build-claude-promotion-smoke-task",
            command="build-claude-promotion-smoke-task",
            description=(
                "Build a prompt, fixture candidate promotion artifact, and "
                "expected-artifact checklist for a Claude Agent SDK canary that "
                "directly executes promotion and strategy-evidence conversion."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_dir",
                "workspace_dir",
                "runtime_contract_path",
                "tool_catalog_path",
            ],
            produced_artifacts=[
                "claude_agent_prompt_md",
                "expected_artifacts_json",
                "candidate_promotion_json",
                "required_command_flow",
            ],
            state_hints=["CLAUDE_PROMOTION_SMOKE_TASK_BUILT"],
            agent_guidance=(
                "Use after the FNO training smoke if no promotable candidate was "
                "produced. This verifies that Claude Agent SDK can drive the "
                "promotion/evidence branch with a known valid fixture rather than "
                "only seeing the tools in the catalog."
            ),
        ),
        ScientificToolSpec(
            name="build-claude-hypothesis-lab-closure-task",
            command="build-claude-hypothesis-lab-closure-task",
            description=(
                "Build a Claude Agent SDK task prompt and expected-artifact "
                "checklist for closing one candidate-family -> Hypothesis Lab "
                "sandbox -> next-policy loop."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_dir",
                "workspace_dir",
                "runtime_contract_path",
                "tool_catalog_path",
                "candidate_family_sandbox_path",
                "context_path",
                "target_path",
            ],
            produced_artifacts=[
                "claude_agent_prompt_md",
                "expected_artifacts_json",
                "required_command_flow",
            ],
            state_hints=["CLAUDE_HYPOTHESIS_LAB_CLOSURE_TASK_BUILT"],
            agent_guidance=(
                "Use this to hand the closure loop to Claude SDK itself. The "
                "generated prompt should be executed by run-claude-agent-sdk-runtime; "
                "Codex should not manually replace the agent for this loop."
            ),
        ),
        ScientificToolSpec(
            name="build-claude-literature-candidate-closure-task",
            command="build-claude-literature-candidate-closure-task",
            description=(
                "Build a Claude Agent SDK task prompt and expected-artifact "
                "checklist for turning a parsed literature artifact into "
                "agent-authored AlgorithmCard v2 candidates, validating them, "
                "compiling candidates, and routing them through the literature "
                "algorithm v2 registry pipeline."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "output_dir",
                "workspace_dir",
                "runtime_contract_path",
                "tool_catalog_path",
                "paper_parse_path",
                "operator_family",
                "registry_path",
                "current_best_score",
            ],
            produced_artifacts=[
                "claude_agent_prompt_md",
                "expected_artifacts_json",
                "agent_algorithm_cards_json",
                "algorithm_card_v2_audit_json",
                "algorithm_candidates_json",
                "literature_registry_json",
            ],
            state_hints=["CLAUDE_LITERATURE_CANDIDATE_CLOSURE_TASK_BUILT"],
            agent_guidance=(
                "Use this after MinerU parsing when candidate space is narrow. "
                "The generated task lets Claude SDK author grounded AlgorithmCard "
                "JSON but still forces validation, compilation, and registry "
                "gates before any strategy can be internalized. Treat output_dir "
                "as a reusable agent workspace; do not create new timestamped "
                "directories for each retry unless the supervisor explicitly asks."
            ),
        ),
        ScientificToolSpec(
            name="build-hypothesis-algorithm-design",
            command="build-hypothesis-algorithm-design",
            description=(
                "Convert failure memories, logs, strategy libraries, and "
                "literature cards into diverse mechanism hypotheses and bounded "
                "AlgorithmBlueprint designs."
            ),
            risk_level="artifact_write",
            required_inputs=["output_path", "evidence_paths"],
            produced_artifacts=[
                "algorithm_design_json",
                "mechanism_hypotheses",
                "algorithm_blueprints",
                "diversity_audit",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_DESIGN_BUILT"],
            agent_guidance=(
                "Use when candidate-family search is saturated or failure memory "
                "shows repeated local parameter replacement. The output must be "
                "validated before sandbox execution; inspect diversity_audit so "
                "the agent does not produce many variants of the same mechanism. Pass "
                "FNO/training evidence with --evidence-path; do not disguise "
                "training evidence as a literature card."
            ),
        ),
        ScientificToolSpec(
            name="validate-hypothesis-algorithm-design",
            command="validate-hypothesis-algorithm-design",
            description=(
                "Validate Hypothesis Lab algorithm designer output for mechanism "
                "diversity, runnable blueprint structure, evaluation plans, and "
                "forbidden-data constraints."
            ),
            risk_level="read_only",
            required_inputs=["design_path", "output_path"],
            produced_artifacts=[
                "algorithm_design_audit_json",
                "violations",
                "decision",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_DESIGN_VALIDATED"],
            agent_guidance=(
                "Run immediately after build-hypothesis-algorithm-design. Only "
                "route accepted designs to sandbox planning; rejected designs need "
                "repair rather than runtime execution."
            ),
        ),
        ScientificToolSpec(
            name="record-hypothesis-algorithm-evidence",
            command="record-hypothesis-algorithm-evidence",
            description=(
                "Internalize sandbox evidence for an AlgorithmBlueprint as an "
                "AlgorithmCard, Evidence, FailureTaxonomy, and next-generation "
                "constraint artifact."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "design_path",
                "sandbox_report_path",
                "output_path",
            ],
            produced_artifacts=[
                "algorithm_evidence_json",
                "algorithm_card",
                "evidence",
                "failure_taxonomy",
                "next_generation_constraints",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_EVIDENCE_RECORDED"],
            agent_guidance=(
                "Run after any sandboxed algorithm blueprint. Feed rejected "
                "mechanism families back into the next designer run so the agent "
                "does not repeat non-improving algorithm mechanisms."
            ),
        ),
        ScientificToolSpec(
            name="plan-hypothesis-algorithm-sandbox",
            command="plan-hypothesis-algorithm-sandbox",
            description=(
                "Route validated AlgorithmBlueprint designs into bounded sandbox "
                "items or blocked executor-authoring tasks."
            ),
            risk_level="artifact_write",
            required_inputs=["design_path", "output_path"],
            produced_artifacts=[
                "algorithm_sandbox_plan_json",
                "sandbox_items",
                "blocked_items",
                "executor_routes",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_SANDBOX_PLANNED"],
            agent_guidance=(
                "Run after validate-hypothesis-algorithm-design accepts a design. "
                "Only execute sandbox_items through their bounded executor_route; "
                "blocked_items require executor authoring or design repair."
            ),
        ),
        ScientificToolSpec(
            name="adapt-hypothesis-algorithm-sandbox-item",
            command="adapt-hypothesis-algorithm-sandbox-item",
            description=(
                "Adapt one Algorithm Designer sandbox item into an existing "
                "controlled Hypothesis Lab blueprint family for bounded execution."
            ),
            risk_level="artifact_write",
            required_inputs=["sandbox_plan_path", "output_path"],
            produced_artifacts=[
                "adapted_blueprint_json",
                "source_algorithm_blueprint_id",
                "candidate_family",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_BLUEPRINT_ADAPTED"],
            agent_guidance=(
                "Use before run-blueprint-sandbox when a designer mechanism has "
                "a supported surrogate executor. This keeps execution inside "
                "existing controlled numpy families instead of arbitrary code."
            ),
        ),
        ScientificToolSpec(
            name="run-hypothesis-algorithm-sandbox",
            command="run-hypothesis-algorithm-sandbox",
            description=(
                "Execute an adapted Algorithm Designer blueprint through the "
                "existing controlled Hypothesis Lab blueprint sandbox."
            ),
            risk_level="sandbox",
            required_inputs=[
                "adapted_blueprint_path",
                "context_path",
                "target_path",
                "output_dir",
            ],
            produced_artifacts=[
                "sandbox_report_path",
                "judge_report_path",
                "best_submission_dir",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_SANDBOX_RAN"],
            agent_guidance=(
                "Use only after planning and adapting a sandbox item. Follow with "
                "record-hypothesis-algorithm-evidence so results affect the next "
                "designer generation."
            ),
        ),
        ScientificToolSpec(
            name="run-hypothesis-algorithm-designer-stability-batch",
            command="run-hypothesis-algorithm-designer-stability-batch",
            description=(
                "Run multiple failure-memory sets through design, validation, "
                "sandbox planning, controlled sandbox execution, and evidence "
                "writeback to measure Hypothesis Lab stability."
            ),
            risk_level="sandbox",
            required_inputs=[
                "memory_set_json",
                "context_path",
                "target_path",
                "output_dir",
            ],
            produced_artifacts=[
                "stability_report_json",
                "design_success_rate",
                "sandbox_success_rate",
                "evidence_writeback_success_rate",
            ],
            state_hints=["HYPOTHESIS_ALGORITHM_DESIGNER_STABILITY_BATCH_RAN"],
            agent_guidance=(
                "Use before trusting a new designer change. Stable output means "
                "multiple failure memories still yield diverse hypotheses and "
                "sandbox evidence; unstable output should drive designer repair."
            ),
        ),
        ScientificToolSpec(
            name="run-hypothesis-learning-constraint-smoke",
            command="run-hypothesis-learning-constraint-smoke",
            description=(
                "Run a no-model smoke from trace KPI to candidate-plan learning "
                "constraints and expansion handoff filtering."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "trace_path",
                "output_dir",
                "local_strategy_library_path",
            ],
            produced_artifacts=[
                "learning_constraint_smoke_json",
                "hypothesis_kpi_report_json",
                "candidate_plan_json",
                "expansion_handoff_report_json",
            ],
            state_hints=["HYPOTHESIS_LEARNING_CONSTRAINT_SMOKE_RAN"],
            agent_guidance=(
                "Use after changing Hypothesis Lab memory, KPI, planner, or "
                "handoff logic. It should prove deprioritized trace events alter "
                "candidate generation before running expensive agent batches."
            ),
        ),
        ScientificToolSpec(
            name="prepare-pde-literature-extraction-skill",
            command="prepare-pde-literature-extraction-skill",
            description="Prepare an agent-facing PDE literature extraction task.",
            risk_level="artifact_write",
            required_inputs=[
                "literature_extraction_context_path",
                "operator_family",
                "target_problem",
            ],
            produced_artifacts=["pde_literature_extraction_skill_json"],
            state_hints=["PDE_LITERATURE_EXTRACTION_SKILL_PREPARED"],
            agent_guidance=(
                "Use after minerU parsing. The output constrains the agent to "
                "extract AlgorithmCard v2 candidates with source-unit grounding."
            ),
        ),
        ScientificToolSpec(
            name="record-agent-algorithm-cards",
            command="record-agent-algorithm-cards",
            description=(
                "Persist agent-authored AlgorithmCard JSON through the controlled "
                "pde-agent CLI instead of direct file writes."
            ),
            risk_level="artifact_write",
            required_inputs=["cards_json", "output_path"],
            produced_artifacts=["algorithm_cards_json"],
            state_hints=["ALGORITHM_CARDS_EXTRACTED"],
            agent_guidance=(
                "Use when the agent has constructed grounded AlgorithmCard v2 "
                "JSON and needs to persist it without free-form Write access. "
                "Treat it as the only supported persistence path for agent-authored "
                "algorithm cards."
            ),
        ),
        ScientificToolSpec(
            name="run-hypothesis-lab",
            command="run-hypothesis-lab",
            description="Run local knowledge-card to blueprint to sandbox pipeline.",
            risk_level="sandbox",
            required_inputs=["context_path", "target_path", "output_dir"],
            produced_artifacts=[
                "knowledge_cards_json",
                "hypotheses_json",
                "blueprints_json",
                "sandbox_report_json",
            ],
            state_hints=["HYPOTHESIS_LAB_RAN", "HYPOTHESIS_LAB_SANDBOX_VERIFIED"],
            agent_guidance=(
                "Use for bounded local experiments. Record the result into the "
                "trace immediately after the run."
            ),
        ),
        ScientificToolSpec(
            name="run-candidate-expansion-handoff",
            command="run-candidate-expansion-handoff",
            description=(
                "Execute an exhausted candidate-plan handoff by building local "
                "asset algorithm candidates and running literature "
                "strategy expansion."
            ),
            risk_level="artifact_write",
            required_inputs=["candidate_plan_path", "workspace_dir"],
            produced_artifacts=[
                "handoff_report_json",
                "algorithm_candidates_json",
                "expansion_report_json",
            ],
            state_hints=["AGENT_CANDIDATE_EXPANSION_HANDOFF_RAN"],
            agent_guidance=(
                "Use when plan-hypothesis-candidates reports "
                "expansion_required. This turns the handoff into concrete "
                "algorithm candidates and records a literature expansion report."
            ),
        ),
        ScientificToolSpec(
            name="run-expansion-sandbox-queue",
            command="run-expansion-sandbox-queue",
            description=(
                "Advance a literature expansion sandbox queue through "
                "short, controlled Hypothesis Lab sandboxes or record explicit "
                "skip outcomes."
            ),
            risk_level="sandbox",
            required_inputs=[
                "expansion_report_path",
                "context_path",
                "target_path",
                "output_dir",
            ],
            produced_artifacts=[
                "queue_report_json",
                "hypothesis_lab_sandbox_reports",
                "failure_memory_json",
            ],
            state_hints=["AGENT_EXPANSION_SANDBOX_QUEUE_RAN"],
            agent_guidance=(
                "Use after run-candidate-expansion-handoff or "
                "the compatibility literature-strategy-expansion path. Only short "
                "Hypothesis Lab compatible plans are executed automatically; "
                "checkpoint and custom executor plans must be skipped with "
                "recorded reason codes unless a separate bounded executor path "
                "has been prepared. For legal-reference refinement retests, pass "
                "selected_algorithm_candidate_id from the candidate refinement "
                "artifact so the agent retests the exact candidate that triggered "
                "the refinement gate."
            ),
        ),
        ScientificToolSpec(
            name="evaluate-expansion-sandbox-queue",
            command="evaluate-expansion-sandbox-queue",
            description=(
                "Convert expansion sandbox queue results into standard sandbox "
                "evidence, apply the AlgorithmCard promotion rubric, and write "
                "candidate outcomes back to the literature registry."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "queue_report_path",
                "output_dir",
                "registry_path",
                "current_best_score",
            ],
            produced_artifacts=[
                "sandbox_evidence_json",
                "candidate_promotion_json",
                "literature_registry_event_json",
                "literature_registry_json",
            ],
            state_hints=["AGENT_EXPANSION_SANDBOX_QUEUE_EVALUATED"],
            agent_guidance=(
                "Use after run-expansion-sandbox-queue. This is the gate that "
                "prevents local sandbox wins from being internalized unless they "
                "also pass the promotion rubric against the declared current best."
            ),
        ),
        ScientificToolSpec(
            name="record-promoted-algorithm-strategy-proposal",
            command="record-promoted-algorithm-strategy-proposal",
            description=(
                "Convert a promoted AlgorithmCard candidate promotion artifact "
                "into the standard strategy-growth proposal artifact that the agent "
                "can rubric, ledger, and skill-promote."
            ),
            risk_level="artifact_write",
            required_inputs=["candidate_promotion_path", "output_path"],
            produced_artifacts=["strategy_proposal_json"],
            state_hints=["STRATEGY_PROPOSAL_RECORDED"],
            agent_guidance=(
                "Use when a literature registry summary reports a "
                "promoted candidate with candidate_promotion_path. This preserves "
                "the candidate grounding, sandbox evidence paths, and promotion "
                "metrics instead of asking the agent to hand-author proposal fields."
            ),
        ),
        ScientificToolSpec(
            name="convert-candidate-promotion-to-strategy-evidence",
            command="convert-candidate-promotion-to-strategy-evidence",
            description=(
                "Convert a candidate promotion artifact into the standard "
                "strategy-growth evidence summary fields consumed by "
                "evaluate-strategy-promotion."
            ),
            risk_level="artifact_write",
            required_inputs=["candidate_promotion_path", "output_path"],
            produced_artifacts=["strategy_evidence_summary_json"],
            state_hints=["STRATEGY_EVIDENCE_SUMMARY_BUILT"],
            agent_guidance=(
                "Use after record-promoted-algorithm-strategy-proposal when "
                "the agent needs a rubric-compatible evidence summary. This keeps "
                "the candidate promotion evidence count intact; if only one "
                "window was tested, the downstream rubric must decide whether "
                "that is sufficient. Pass repeated "
                "--extra-candidate-promotion-path values to accumulate multiple "
                "legal-reference retests for the same candidate."
            ),
        ),
        ScientificToolSpec(
            name="plan-candidate-evidence-accumulation",
            command="plan-candidate-evidence-accumulation",
            description=(
                "Plan additional legal-reference retests when strategy-growth "
                "rejects a promoted candidate only because evidence windows are "
                "insufficient."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "strategy_promotion_report_path",
                "output_path",
            ],
            produced_artifacts=["evidence_accumulation_plan_json"],
            state_hints=["CANDIDATE_EVIDENCE_ACCUMULATION_PLANNED"],
            agent_guidance=(
                "Use after evaluate-strategy-promotion returns "
                "insufficient_evaluation_windows. The plan states how many "
                "additional legal-reference retests are needed before rerunning "
                "convert-candidate-promotion-to-strategy-evidence with the "
                "accumulated promotion paths."
            ),
        ),
        ScientificToolSpec(
            name="prepare-expansion-blocked-executor-authoring-tasks",
            command="prepare-expansion-blocked-executor-authoring-tasks",
            description=(
                "Turn blocked expansion sandbox queue items into bounded "
                "executor design specs, scaffolded executor templates, and "
                "agent authoring tasks."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "queue_report_path",
                "output_dir",
                "context_path",
                "target_path",
            ],
            produced_artifacts=[
                "executor_design_request_json",
                "bounded_executor_design_spec_json",
                "bounded_executor_template_json",
                "bounded_executor_authoring_task_json",
            ],
            state_hints=["AGENT_EXPANSION_BLOCKED_EXECUTOR_TASKS_PREPARED"],
            agent_guidance=(
                "Use when a registry summary reports executor_gap_blocks_algorithm_family. "
                "This prepares a constrained edit task; it does not authorize "
                "running the original heavy CLI command or arbitrary external code."
            ),
        ),
        ScientificToolSpec(
            name="audit-bounded-executor-authoring-task",
            command="audit-bounded-executor-authoring-task",
            description=(
                "Audit a bounded executor authoring task before handing it to "
                "an agent for implementation."
            ),
            risk_level="read_only",
            required_inputs=["authoring_task_path", "skill_root", "output_path"],
            produced_artifacts=["bounded_executor_authoring_audit_json"],
            state_hints=["BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED"],
            agent_guidance=(
                "Use after prepare-expansion-blocked-executor-authoring-tasks "
                "and before agent-author-bounded-executor. Proceed only when "
                "decision is approved; rejected audits mean the task contract "
                "or skill path must be regenerated."
            ),
        ),
        ScientificToolSpec(
            name="plan-candidate-sandbox",
            command="plan-candidate-sandbox",
            description="Create a bounded sandbox plan for one algorithm candidate.",
            risk_level="artifact_write",
            required_inputs=["algorithm_candidate_path", "output_path"],
            produced_artifacts=["sandbox_plan_json"],
            state_hints=["ALGORITHM_CANDIDATE_SANDBOX_PLANNED"],
            agent_guidance=(
                "Use to convert a candidate into an executable experiment plan "
                "before any implementation work."
            ),
        ),
        ScientificToolSpec(
            name="run-bounded-executor-sandbox",
            command="run-bounded-executor-sandbox",
            description="Run a bounded executor template against a sandbox fixture.",
            risk_level="sandbox",
            required_inputs=[
                "design_spec_path",
                "manifest_path",
                "executor_path",
                "context_path",
                "target_path",
                "output_dir",
            ],
            produced_artifacts=[
                "bounded_executor_sandbox_report_json",
                "prediction_hdf5",
            ],
            state_hints=["BOUNDED_EXECUTOR_SANDBOX_RAN"],
            agent_guidance=(
                "Use only for executor implementations that passed validation. "
                "Record pass/fail and reason codes into the literature registry."
            ),
        ),
        ScientificToolSpec(
            name="record-bounded-executor-sandbox-evidence",
            command="record-bounded-executor-sandbox-evidence",
            description=(
                "Convert a bounded executor sandbox report into standard "
                "AlgorithmCard promotion evidence."
            ),
            risk_level="artifact_write",
            required_inputs=[
                "sandbox_report_path",
                "output_path",
                "current_best_score",
            ],
            produced_artifacts=["sandbox_evidence_json"],
            state_hints=["SANDBOX_EVIDENCE_RECORDED"],
            agent_guidance=(
                "Use after run-bounded-executor-sandbox and before "
                "evaluate-algorithm-candidate-promotion so negative bounded "
                "executor results are deprioritized by the same rubric as "
                "other Hypothesis Lab candidates."
            ),
        ),
        ScientificToolSpec(
            name="complete-bounded-executor-authoring-task",
            command="complete-bounded-executor-authoring-task",
            description=(
                "After an agent edits executor.py, run the bounded executor "
                "validation, sandbox, evidence, promotion, and registry loop."
            ),
            risk_level="sandbox",
            required_inputs=[
                "authoring_task_path",
                "algorithm_candidates_path",
                "registry_path",
                "current_best_score",
            ],
            produced_artifacts=[
                "bounded_executor_audit_json",
                "bounded_executor_sandbox_report_json",
                "sandbox_evidence_json",
                "candidate_promotion_json",
                "literature_registry_json",
                "hypothesis_lab_trace_event",
            ],
            state_hints=["BOUNDED_EXECUTOR_AUTHORING_COMPLETED"],
            agent_guidance=(
                "Use only after agent-author-bounded-executor has edited the "
                "allowed executor.py. This command does not author code; it "
                "audits, measures, applies promotion rubric, and writes failure "
                "or promotion evidence back to the registry. Pass trace_path "
                "when available so the agent can learn from the completion "
                "decision in later iterations."
            ),
        ),
    ]
    if include_compatibility_tools:
        return specs
    return [spec for spec in specs if not _is_legacy_tool(spec)]


def build_scientific_tool_catalog(
    *,
    output_path: str | Path | None,
    force: bool,
    include_compatibility_tools: bool = False,
) -> dict[str, Any]:
    output_file = Path(output_path) if output_path is not None else None
    _guard_output(output_file, force=force)
    tools = [
        spec.to_dict()
        for spec in default_scientific_tool_specs(
            include_compatibility_tools=include_compatibility_tools
        )
    ]
    catalog = {
        "schema_version": 1,
        "state_hint": STATE_HINT,
        "primary_agent_runtime": "claude_agent_sdk",
        "legacy_agent_runtimes": ["legacy"] if include_compatibility_tools else [],
        "runtime_policy": {
            "active_substrate": "claude_agent_sdk",
            "legacy_substrates": ["legacy"] if include_compatibility_tools else [],
            "legacy_tool_visibility": (
                "hidden unless include_compatibility_tools is explicitly enabled"
            ),
            "active_task1_training_entrypoint": "run-claude-agent-sdk-runtime",
        },
        "compatibility_tools_included": include_compatibility_tools,
        "tool_count": len(tools),
        "tools": tools,
        "risk_levels": sorted(VALID_RISK_LEVELS),
        "next_command": "summarize-hypothesis-lab-trace",
    }
    _write_optional_json(output_file, catalog)
    return catalog


def _is_legacy_tool(spec: ScientificToolSpec) -> bool:
    if spec.runtime_status == "deprecated_compatibility":
        return True
    return "legacy" in spec.name.lower() or "legacy" in spec.command.lower()


def _guard_output(path: Path | None, *, force: bool) -> None:
    if path is not None and path.exists() and not force:
        raise ScientificToolRouterError(f"output already exists: {path}")


def _write_optional_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
