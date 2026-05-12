"""Literature algorithm lab constants and configuration."""

from __future__ import annotations

SOURCE_RECORDS_STATE_HINT = "LITERATURE_SOURCE_RECORDS_BUILT"
SOURCE_SEARCH_STATE_HINT = "EXTERNAL_SOURCE_SEARCH_NORMALIZED"
SOURCE_DOCUMENT_STATE_HINT = "SOURCE_DOCUMENT_DOWNLOADED"
MINERU_TASK_SUBMITTED_STATE_HINT = "MINERU_PARSE_TASK_SUBMITTED"
MINERU_TASK_POLLED_STATE_HINT = "MINERU_PARSE_TASK_POLLED"
MINERU_RESULT_DOWNLOADED_STATE_HINT = "MINERU_PARSE_RESULT_DOWNLOADED"
PAPER_PARSE_STATE_HINT = "MINERU_PARSE_INGESTED"
ALGORITHM_CARDS_STATE_HINT = "ALGORITHM_CARDS_EXTRACTED"
ALGORITHM_CARDS_V2_VALIDATED_STATE_HINT = "ALGORITHM_CARDS_V2_VALIDATED"
ALGORITHM_CANDIDATES_STATE_HINT = "ALGORITHM_CANDIDATES_COMPILED"
ALGORITHM_CANDIDATE_PROMOTION_STATE_HINT = "ALGORITHM_CANDIDATE_PROMOTION_EVALUATED"
LITERATURE_REGISTRY_EVENT_STATE_HINT = "LITERATURE_REGISTRY_EVENT_RECORDED"
EXECUTABLE_BLUEPRINTS_VALIDATED_STATE_HINT = "EXECUTABLE_BLUEPRINTS_VALIDATED"
SANDBOX_PLAN_STATE_HINT = "ALGORITHM_CANDIDATE_SANDBOX_PLANNED"
EXECUTOR_TAXONOMY_STATE_HINT = "EXECUTOR_TAXONOMY_BUILT"
FNO_CANDIDATE_ADAPTED_STATE_HINT = "FNO_ARCHITECTURE_CANDIDATE_ADAPTED"
EXECUTOR_DESIGN_REQUEST_STATE_HINT = "EXECUTOR_DESIGN_REQUEST_CREATED"
BLOCKED_CANDIDATE_HANDOFF_STATE_HINT = "BLOCKED_CANDIDATE_HANDOFF_BUILT"
LITERATURE_REGISTRY_UPDATED_STATE_HINT = "LITERATURE_REGISTRY_UPDATED"
LITERATURE_ALGORITHM_V2_PIPELINE_STATE_HINT = "LITERATURE_ALGORITHM_V2_PIPELINE_RAN"
ARXIV_MINERU_PARSE_PIPELINE_STATE_HINT = "ARXIV_MINERU_PARSE_PIPELINE_RAN"
LITERATURE_EXTRACTION_CONTEXT_STATE_HINT = "LITERATURE_EXTRACTION_CONTEXT_PREPARED"
FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT = "FNO_LITERATURE_ADAPTATION_TASK_PREPARED"
DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT = (
    "DEEPONET_LITERATURE_ADAPTATION_TASK_PREPARED"
)
UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT = "UNET_LITERATURE_ADAPTATION_TASK_PREPARED"
BOUNDED_EXECUTOR_DESIGN_SPEC_STATE_HINT = "BOUNDED_EXECUTOR_DESIGN_SPEC_CREATED"
BOUNDED_EXECUTOR_TEMPLATE_STATE_HINT = "BOUNDED_EXECUTOR_TEMPLATE_MATERIALIZED"
BOUNDED_EXECUTOR_AUDIT_STATE_HINT = "BOUNDED_EXECUTOR_IMPLEMENTATION_VALIDATED"
BOUNDED_EXECUTOR_SANDBOX_STATE_HINT = "BOUNDED_EXECUTOR_SANDBOX_RAN"
BOUNDED_EXECUTOR_AUTHORING_TASK_STATE_HINT = "BOUNDED_EXECUTOR_AUTHORING_TASK_PREPARED"
BOUNDED_EXECUTOR_AUTHORING_AUDIT_STATE_HINT = (
    "BOUNDED_EXECUTOR_AUTHORING_TASK_AUDITED"
)
BOUNDED_EXECUTOR_AUTHORING_COMPLETION_STATE_HINT = (
    "BOUNDED_EXECUTOR_AUTHORING_COMPLETED"
)
VALID_SOURCE_TYPES = {
    "openalex_work",
    "arxiv_paper",
    "github_repo",
    "competition_resource",
    "manual_source",
}
MINERU_AGENT_URL_ENDPOINT = "https://mineru.net/api/v1/agent/parse/url"
MINERU_AGENT_POLL_ENDPOINT = "https://mineru.net/api/v1/agent/parse"
MINERU_PRECISE_URL_ENDPOINT = "https://mineru.net/api/v4/extract/task"
HOOK_TO_FAMILY = {
    "official_checkpoint_calibration": {
        "candidate_family": "official_checkpoint_family",
        "model_family": "checkpoint",
        "cli_command": "run-checkpoint-family-sandbox",
        "sandbox_level": 1,
    },
    "learned_residual_correction": {
        "candidate_family": "lightweight_residual_corrector",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "spectral_rollout_stabilization": {
        "candidate_family": "spectral_rollout_stabilization",
        "model_family": "postprocess",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 0,
    },
    "physics_residual_regularization": {
        "candidate_family": "physics_residual_regularization",
        "model_family": "FNO",
        "cli_command": "run-blueprint-sandbox",
        "sandbox_level": 1,
    },
    "scheduled_rollout_training": {
        "candidate_family": "scheduled_rollout_training",
        "model_family": "FNO",
        "cli_command": "execute-fno-finetune-candidate",
        "sandbox_level": 1,
    },
    "operator_correction_postprocess": {
        "candidate_family": "operator_correction_postprocess",
        "model_family": "postprocess",
        "cli_command": "run-hypothesis-lab",
        "sandbox_level": 0,
    },
}
HOOK_ALIASES = {
    "physics_informed_loss": "physics_residual_regularization",
    "pinn_residual_loss": "physics_residual_regularization",
    "pi_deeponet_physics_loss": "physics_residual_regularization",
    "scheduled_sampling": "scheduled_rollout_training",
    "rollout_training": "scheduled_rollout_training",
    "autoregressive_rollout_training": "scheduled_rollout_training",
    "postprocess_operator_correction": "operator_correction_postprocess",
    "residual_blend_postprocess": "operator_correction_postprocess",
}
HOOK_TO_ALGORITHM_CARD_V2 = {
    "official_checkpoint_calibration": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "checkpoint",
        "mathematical_object": "checkpoint_weights",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": True,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "learned_residual_correction": {
        "operator_target": "residual_to_correction",
        "mechanism_type": "postprocess",
        "mathematical_object": "residual_operator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": True,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "spectral_rollout_stabilization": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "spectral_filter",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": False,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "physics_residual_regularization": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "loss",
        "mathematical_object": "burgers_residual",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "scheduled_rollout_training": {
        "operator_target": "context_to_rollout",
        "mechanism_type": "rollout",
        "mathematical_object": "time_integrator",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
    "operator_correction_postprocess": {
        "operator_target": "postprocess_to_solution",
        "mechanism_type": "postprocess",
        "mathematical_object": "correction_operator",
        "implementation_requirements": {
            "requires_training": False,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": False,
            "requires_external_code": False,
            "can_run_as_postprocess": True,
        },
    },
    "fourier_neural_operator_architecture": {
        "operator_target": "ic_to_trajectory",
        "mechanism_type": "architecture",
        "mathematical_object": "fourier_modes",
        "implementation_requirements": {
            "requires_training": True,
            "requires_checkpoint": False,
            "requires_target_data": True,
            "requires_new_executor": True,
            "requires_external_code": False,
            "can_run_as_postprocess": False,
        },
    },
}
ALGORITHM_CARD_V2_OPERATOR_TARGETS = {
    "ic_to_trajectory",
    "coefficient_to_solution",
    "residual_to_correction",
    "context_to_rollout",
    "postprocess_to_solution",
}
ALGORITHM_CARD_V2_MECHANISM_TYPES = {
    "architecture",
    "loss",
    "rollout",
    "postprocess",
    "conditioning",
    "checkpoint",
    "hybrid",
}
ALGORITHM_CARD_V2_EVIDENCE_STATUSES = {
    "extracted",
    "candidate_compiled",
    "sandbox_planned",
    "tested",
    "promoted",
    "rejected",
    "blocked",
}
ALGORITHM_CARD_V2_IMPLEMENTATION_REQUIREMENTS = {
    "requires_training",
    "requires_checkpoint",
    "requires_target_data",
    "requires_new_executor",
    "requires_external_code",
    "can_run_as_postprocess",
}
FAILURE_TAXONOMY = {
    "source_download_failed",
    "parse_failed",
    "parse_quality_insufficient",
    "no_algorithmic_claim",
    "unsupported_executor",
    "data_contract_incompatible",
    "requires_external_code",
    "candidate_schema_invalid",
    "blueprint_schema_invalid",
    "sandbox_rule_failed",
    "context_mismatch",
    "target_leakage_risk",
    "score_regressed",
    "runtime_budget_exceeded",
    "physics_regressed",
    "ablation_missing_or_weak",
    "insufficient_reproducibility",
    "runtime_exceeded",
    "duplicate_family",
    "insufficient_evidence",
    "reproducibility_failed",
}
EXECUTABLE_BLUEPRINT_BASELINES = {
    "persistence",
    "fno_checkpoint",
    "retrieval_prior",
}
EXECUTABLE_BLUEPRINT_TRANSFORMS = {
    "linear_trend",
    "spectral_smooth",
    "residual_fit",
    "retrieval_prior",
    "retrieval_refinement",
    "checkpoint_calibration",
}
EXECUTABLE_BLUEPRINT_TRAINING_MODES = {
    "none",
    "finetune",
    "supervised",
    "physics_loss",
}
EXECUTABLE_BLUEPRINT_EXECUTORS = {
    "controlled_numpy_operator",
    "controlled_numpy_residual_operator",
    "controlled_numpy_retrieval_prior",
    "controlled_numpy_retrieval_refinement",
    "controlled_numpy_task1_early_refinement",
    "controlled_numpy_sem_burgers_rollout",
    "controlled_numpy_burgers_residual_refinement",
    "controlled_numpy_pi_deeponet_correction_head",
    "official_checkpoint_candidate_pipeline",
}
EXECUTOR_DESIGN_HOOKS = {
    "fourier_neural_operator_architecture": {
        "route": "requires_executor_design",
        "requested_executor_id": "bounded_fno_architecture_executor",
        "fallback_route": "official_checkpoint_family",
        "rationale": (
            "A paper-level FNO architecture claim needs bounded local implementation "
            "or an official-checkpoint adapter before sandbox execution."
        ),
    }
}
OPERATOR_LITERATURE_ADAPTATION_PROFILES: dict[str, dict[str, Any]] = {
    "fno": {
        "state_hint": FNO_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "fno_literature_adaptation_task_path",
        "operator_family": "FNO",
        "skill_name": "pde-fno-literature-adapter",
        "next_command": "agent-extract-fno-algorithm-cards",
        "keywords": [
            "fourier",
            "spectral",
            "operator",
            "convolution",
            "mode",
            "resolution",
            "burgers",
        ],
        "reading_focus": [
            "spectral convolution and Fourier mode truncation",
            "lifting and projection around Fourier layers",
            "resolution or discretization generalization claims",
            "Burgers-specific rollout stability and high-frequency behavior",
            "training rollout, checkpoint, or calibration procedures",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "spectral_rollout_stabilization",
                "official_checkpoint_calibration",
                "physics_residual_regularization",
                "scheduled_rollout_training",
            ],
            "requires_bounded_executor": [
                "fourier_neural_operator_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "ic_to_trajectory",
                "context_to_rollout",
                "postprocess_to_solution",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "rollout",
                "postprocess",
                "checkpoint",
                "loss",
            ],
            "required_observables": [
                "local judge score or segment score",
                "late-horizon RMSE or spectral-energy stability",
                "runtime under task budget",
                "first-context-step identity preservation",
            ],
        },
        "task": (
            "Use FNO domain knowledge to turn grounded context units into "
            "AlgorithmCard v2 records without fabricating claims."
        ),
    },
    "deeponet": {
        "state_hint": DEEPONET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "deeponet_literature_adaptation_task_path",
        "operator_family": "DeepONet",
        "skill_name": "pde-deeponet-literature-adapter",
        "next_command": "agent-extract-deeponet-algorithm-cards",
        "keywords": [
            "deeponet",
            "deep operator",
            "branch",
            "trunk",
            "sensor",
            "coordinate",
            "pi-deeponet",
            "physics-informed",
            "operator",
            "burgers",
        ],
        "reading_focus": [
            "branch network encoding of input functions or sensors",
            "trunk network coordinate basis and branch-trunk merge rule",
            "PI-DeepONet residual or physics-informed loss terms",
            "rollout construction from pointwise operator evaluations",
            "Burgers context-window conditioning and target coordinate layout",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "physics_residual_regularization",
                "scheduled_rollout_training",
                "operator_correction_postprocess",
            ],
            "requires_bounded_executor": [
                "deeponet_branch_trunk_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "ic_to_trajectory",
                "context_to_rollout",
                "coordinate_to_solution",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "loss",
                "conditioning",
                "rollout",
                "hybrid",
            ],
            "required_observables": [
                "local judge score or segment score",
                "coordinate-grid reconstruction error",
                "physics residual when a residual loss is claimed",
                "runtime under task budget",
                "first-context-step identity preservation",
            ],
        },
        "task": (
            "Use DeepONet and PI-DeepONet domain knowledge to turn grounded "
            "context units into AlgorithmCard v2 records without fabricating claims."
        ),
    },
    "unet": {
        "state_hint": UNET_LITERATURE_ADAPTATION_TASK_STATE_HINT,
        "path_key": "unet_literature_adaptation_task_path",
        "operator_family": "UNet",
        "skill_name": "pde-unet-literature-adapter",
        "next_command": "agent-extract-unet-algorithm-cards",
        "keywords": [
            "u-net",
            "unet",
            "encoder",
            "decoder",
            "skip",
            "convolution",
            "rollout",
            "frame",
            "checkpoint",
            "burgers",
        ],
        "reading_focus": [
            "encoder-decoder hierarchy and skip connection mechanism",
            "temporal frame prediction, push-forward, or autoregressive rollout",
            "checkpoint, finetuning, or calibration procedures",
            "stability controls for long-horizon generated frames",
            "Burgers context-window conditioning and output-frame semantics",
            "minimal sandbox experiment implied by the paper mechanism",
        ],
        "recommended_hooks": {
            "existing_executor": [
                "official_checkpoint_calibration",
                "scheduled_rollout_training",
                "operator_correction_postprocess",
            ],
            "requires_bounded_executor": [
                "unet_encoder_decoder_architecture",
            ],
        },
        "algorithm_card_guidance": {
            "preferred_operator_targets": [
                "context_to_rollout",
                "postprocess_to_solution",
                "frame_to_frame",
            ],
            "preferred_mechanism_types": [
                "architecture",
                "rollout",
                "checkpoint",
                "postprocess",
                "hybrid",
            ],
            "required_observables": [
                "local judge score or segment score",
                "late-horizon RMSE or drift",
                "runtime under task budget",
                "context preservation",
                "failure evidence for oversmoothing or unstable rollout",
            ],
        },
        "task": (
            "Use UNet operator-family knowledge to turn grounded context units "
            "into AlgorithmCard v2 records without fabricating claims."
        ),
    },
}
