"""
Gender misalignment + semantic gender direction experiment.

This package is standalone and does NOT depend on the original
Model Organisms / EM repo. It is tailored to:

- Misaligned model: ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train
- Optional aligned base: unsloth/qwen2.5-14B-Instruct
- Judge: Qwen/Qwen2.5-32B-Instruct
- Gender semantics: WinoBias (Elfsong/Wino_Bias)
- Domain-specific misalignment: sexism on open-ended gender questions
"""
