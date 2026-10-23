# Source and implementation audit — 2026-10-23

Official Hugging Face API metadata was retrieved on 2026-10-23:

* Dataset: https://huggingface.co/datasets/mib-bench/ioi — MIT, revision
  `e5f3468f3af4c0883be35cd3bced8c711c95d286`.
  https://huggingface.co/api/datasets/mib-bench/ioi
* Model: https://huggingface.co/openai-community/gpt2 — MIT, revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
  https://huggingface.co/api/models/openai-community/gpt2
* Official train JSON at that dataset revision was read: 10,000 records.
  The actual field is `s2_io_flip_counterfactual`, a nested object containing
  prompt, choices, and answerKey. Answer keys are integer indices.
  The first three unchanged source records are test fixtures; no records were created.

The official dataset and model cards at those exact revisions were read and
retained in reports/upstream/. Both declare `license: mit`. The MIB code
repository LICENSE was also retrieved and read: Apache License 2.0, retained
as tests/fixtures/MIB_CODE_LICENSE.txt with provenance in SOURCE.md.

Installed API audit: TransformerLens 4.2.0, datasets 5.1.0, transformers 5.20.2.
Inspected actual signatures and installed source before integration. In 4.2.0,
`boot_transformers` accepts revision/device/dtype; source documents eager HF
attention and hook_z reshaping to [batch, seq, n_heads, d_head]. The runtime
checks must verify those semantics on the pretrained model; they have not run.
The HF API endpoint `/api/models/gpt2/revision/607a30d783dfa663caf39e06633721c8d4cfcd7e`
was also checked: it resolves the literal `gpt2` boot identifier to
`openai-community/gpt2` at the same pinned hash used by the reference loader.

Dataset source JSON is not redistributed in full. Preserve upstream licenses
when redistributing data/model artifacts. This repository does not assert a new
license over upstream data or model weights.

Population rule: SHA-256 rank source-row identities within template, then
round-robin over lexicographically sorted templates until 1,024 are selected.
User delegated this previously unspecified rule on 2026-10-23. No outcomes enter
the rule. Source index plus dataset revision identifies a record. Smoke and
precision checks are separate diagnostic subsets, never confirmation results.

Numerical tolerances in config are initial engineering gates, not measured
agreement or scientific MESOIs. Failures stop caching and require investigation;
do not relax thresholds after looking at effects without recording the change.

Novelty audit and statistical inference are outside this implementation step.
No theoretical novelty or GO decision is claimed. Standard top-k polyhedral
conditioning is established, as stated in SPEC.md.
