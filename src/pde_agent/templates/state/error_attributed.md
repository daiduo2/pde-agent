Current state: ERROR_ATTRIBUTED

Allowed actions:
- read manifest summary
- read rollout metrics
- read spectrum metrics
- read physics check metrics

Forbidden actions:
- patch code
- run training
- propose more than one primary attribution

Required output fields:
- primary_attribution
- secondary_attribution
- evidence
- counterevidence

Hard rules:
- Do not infer missing paths.
- Do not summarize before reading required JSON fields.
- Do not propose more than one primary hypothesis.
