PROJECT TITLE:
Selection-Trace-Conditioned Circuit Inference (STCCI):
Valid Inference After Adaptive Neural Circuit Discovery

ROLE AND OBJECTIVE:
Implement a reproducible PyTorch research project testing
whether adaptive neural circuit discovery inflates causal
intervention-effect estimates, and whether selection-aware
inference improves on established valid methods.

Do not optimize benchmark accuracy.
Do not create or annotate data.
Do not claim discovery of a ground-truth neural mechanism.
Never report an experiment as executed unless it was run.

HYPOTHESIS (falsifiable, verbatim):
"Circuit importance estimates selected and evaluated on
overlapping examples exhibit systematic selection
inflation; independently evaluated effect estimates and
selection-aware confidence sets reduce false circuit
discoveries while preserving the strongest reproducible
causal effects."

EXPLANATORY MECHANISM:
Adaptive search preferentially selects components with
positive estimation noise.

Because intervention effects across components are
correlated, the resulting selection bias depends on both
the component search rule and their joint sampling
distribution.

Conditioning on the recorded selection event may permit
valid reuse of discovery observations while retaining
more estimation information than honest sample splitting.

The claim concerns statistical intervention effects,
not the uniquely correct native computation of the model.

METHOD:
Name: Selection-Trace-Conditioned Circuit Inference.

OBJECTIVE / EQUATIONS:

1. Freeze a language model f and a counterfactual
   activation-patching procedure I_e for each component e.

2. For input pair x_i, define the reference corrupted
   model output and a head-patched model output.

   Let a_i and b_i be the clean target and distractor
   answer-token coordinates.

   Define the bounded score:

     m(z_i) = tanh((z_i[a_i] - z_i[b_i]) / 2)

   Define the per-input intervention effect:

     Y[i,e] = m(f(I_e(x_i))) - m(f(x_i_corrupt))

   Thus Y[i,e] lies in [-2, 2].

   The bounded score is the prespecified estimand.
   Do not change to raw logit difference after results
   are observed.

3. Let P_N be the uniform empirical distribution over
   N fixed, existing, eligible input pairs.

   The finite-population intervention effect is:

     tau[e] = (1/N) sum_i Y[i,e]

4. Given discovery observations D, compute:

     s[e] = mean_{i in D} Y[i,e]

   Select the k highest positive-oriented scores, using
   a deterministic tie-breaking rule.

   The selected set S is shared by every inference method.

5. For selection event S, the top-k inequalities are:

     s[j] - s[i] <= 0
     for all i in S, j not in S.

   This defines a polyhedral selection region.

6. Let Z contain the discovery-sample score vector and
   full-sample effect estimators.

   Under an appropriate multivariate normal approximation:

     Z ~ N(mu, V)

   and the selection event can be represented as:

     A Z <= b.

7. For selected effect c^T mu, decompose:

     T = c^T Z
     v = V c / (c^T V c)
     W = Z - v T.

   Condition on W and the selection event.

   The inequalities A(W + v T) <= b define an interval
   or intersection of intervals for T.

   Invert the resulting truncated-normal distribution
   to construct selection-adjusted confidence intervals.

   IMPORTANT:
   Polyhedral Gaussian selective inference is established.
   Implementing this top-k case is validation, not a new
   theoretical contribution.

   Estimated covariance makes finite-sample exactness
   unavailable unless additional assumptions are proved.
   Explicitly test the approximation's coverage.

8. Research extension:
   Implement sequential greedy circuit selection, where
   the next candidate is evaluated in the context of the
   current selected circuit C.

   Define the contextual incremental effect:

     Y[C,e](x) = Y[C union {e}](x) - Y[C](x)

   Record every candidate comparison at each selected
   search step.

   Investigate a non-enumerative representation of the
   resulting selection-trace event and a tractable
   conditional inference procedure.

   The novelty target is a theorem, counterexample or
   efficiency result specific to adaptive circuit-dependent
   selection. Do not claim that standard polyhedral
   conditioning is novel.

ALGORITHM (PSEUDOCODE):

PHASE 0 — NOVELTY GATE
  Review relevant selective-inference and circuit papers.
  Write down the claimed new theorem or counterexample.
  Identify its assumptions and why existing results do
  not already imply it.
  If this cannot be established, mark NOVELTY_FAILURE.

PHASE 1 — DATA AND MODEL
  Load official MIB IOI training records.
  Select the supplied s2_io_flip counterfactual field.
  Validate answer choices and counterfactual metadata.
  Keep records with unambiguous compatible answer tokens.
  Log all excluded records and reasons.
  Fix N=1024 eligible records by a frozen, template-aware
  sampling rule.
  Load frozen GPT-2 Small.
  Validate clean/corrupted forward outputs.

PHASE 2 — EFFECT CACHE
  Define a fixed set of 24 attention heads.
  Cache clean head-output activations.
  For each input/head, patch the last prompt-token
  attention-head output into the corrupted forward pass.
  Calculate bounded Y[i,e] in fp32.
  Save the N x 24 effect matrix with metadata.
  Compute tau[e] as the exact finite-population reference.
  Seal tau from all selection and interval-fitting code.

PHASE 3 — SELECTIVE INFERENCE
  For each random sampling seed:
    Draw 256 indices independently with replacement
    from the fixed N-record empirical population.

    Use first 128 draws for discovery.
    Use remaining 128 as independent inference draws.

    Select the same top 4 heads for all methods, using
    only the discovery half.

    NAIVE:
      Estimate selected effects on discovery observations
      and compute ordinary unadjusted intervals.

    HONEST:
      Estimate selected effects on the independent
      inference half, with family-wise adjustment.

    BONFERRONI / MAX-T:
      Construct simultaneous intervals for the
      fixed candidate family using prespecified procedures.

    STCCI:
      Estimate selected effects using all 256 draws.
      Account for selection on the first half through
      the joint covariance and selection inequalities.
      Construct selection-conditioned intervals with
      family-wise adjustment.

    Record the same selected heads for every method.
    Save intervals, widths, selected effects, and seeds.

PHASE 4 — STATISTICAL EVALUATION
  Compare all intervals with sealed finite-population tau.
  Calculate coverage and mean interval width.
  Calculate same-sample selection inflation.
  Run null, randomized-selection and covariance-ablation
  controls.
  Estimate replication variance.
  Perform confidence intervals, power calculations
  and TOST.
  Produce GO / KILL / INCONCLUSIVE.

PHASE 5 — RESEARCH EXTENSION, ONLY IF JUSTIFIED
  Replace top-k with sequential contextual selection.
  Derive selection-trace constraints.
  Prove or refute claimed validity/efficiency properties.
  Expand to more candidates and existing public inputs.
  Recompute budgets before running.
  Keep the original hypothesis and estimand fixed.

HYPERPARAMETERS AND DEFAULTS:
  seed_smoke_tests: [11, 23, 37, 41, 53]
  calibration_resampling_seeds: 100 independent draws
  confirmation_resampling_seeds: 1000 independent draws
  initial_population_N: 1024 existing input pairs
  selection_draws: 128
  independent_inference_draws: 128
  total_draws_per_replication: 256
  sampling: with replacement from empirical P_N
  candidate_heads: 24
  candidate_rule: heads 0 and 1 in each of 12 layers
  selected_heads: 4
  selection_rule: largest signed mean effect
  selection_tiebreak: ascending layer/head index
  attention_position: final prompt token
  maximum_sequence_length: 128
  batch_size: 8
  familywise_alpha: 0.05
  covariance: full sample covariance, no shrinkage initially
  covariance_stabilization: prespecified numerical jitter
  inference_arithmetic: fp64 on CPU where feasible
  model_forward: fp16, fp32 confirmation for sensitive effects
  no training or optimizer
  save one row per observed intervention result
  freeze all configs before confirmatory evaluation

The five listed seeds are a minimum smoke-test set.
The 100 calibration and 1000 confirmation draws must use
disjoint RNG streams.

Conditional on the fixed empirical population, independent
with-replacement sampling streams constitute independent
Monte Carlo experiments. They are NOT independent
pretrained-model seeds or independent new-data populations.

DATASETS (name, URL, license, subset/split):

PRIMARY:
  MIB — Indirect Object Identification
  https://huggingface.co/datasets/mib-bench/ioi
  Dataset license: MIT.
  Access: public, no approval.
  Use: existing train split, 10,000 source records;
       select 1024 eligible existing paired records.
  Required fields:
    prompt, choices, answerKey, template,
    supplied counterfactual prompt/choices/answerKey.

  Follow official field schemas instead of assuming all
  counterfactual fields have identical answer keys.

  Do not combine train/dev/test while fitting inference
  procedures or selecting hyperparameters.

SECONDARY:
  MIB official validation and public-test splits.
  Same official dataset link and license.
  Reserved for transfer/robustness assessment.
  Do not use private leaderboard test data.

Dataset project:
  https://github.com/aaronmueller/MIB
  Repository code license: Apache 2.0.

MODELS (name, checkpoint link, regime, precision):

PRIMARY:
  GPT-2 Small, approximately 124M parameters
  https://huggingface.co/openai-community/gpt2
  License: MIT.
  Regime: frozen inference-only.
  Precision: fp16 forward; fp32 logit/effect accumulation;
             fp32 re-evaluation for numerical checks.

IMPLEMENTATION LIBRARY:
  PyTorch and TransformerLens 4.x.

  Use:
    from transformer_lens.model_bridge import TransformerBridge

  Do not use the removed HookedTransformer class in
  TransformerLens 4.x.

  Use TransformerBridge.boot_transformers("gpt2") and
  verify model outputs against reference Hugging Face GPT-2.

  Inspect registered hook names/shapes. Use the
  attention-head output hook_z or its documented alias.
  Test the hook path before caching interventions.

  Pin tested versions and checkpoint revision hashes.

OPTIONAL SCALE CHECK:
  Pythia-160M or a public 0.5B model, only after
  compatibility and task behavior are independently
  validated; this is not required for the kill test.

BASELINES (with citations and links):

1. Naive same-sample confidence intervals.
   Expected to be vulnerable to selection bias.
   Diagnostic only; not the strongest baseline.

2. Honest sample splitting.
   Select using the first 128 draws; infer effects using
   the independent second 128 draws.
   Mandatory valid baseline.

3. Bonferroni simultaneous intervals over the entire
   fixed candidate family.
   Established family-wise correction.

4. Covariance-aware max-t simultaneous inference.
   Use an appropriate multiplier/bootstrap procedure and
   validate coverage under the specified sampling model.

5. Top-k attribution/circuit selection motivation:
   Syed, Rager and Conmy, BlackboxNLP 2024:
   https://aclanthology.org/2024.blackboxnlp-1.25/

6. Variance and circuit-instability baseline:
   Meloux et al., ICML 2026:
   https://proceedings.mlr.press/v306/meloux26a.html

7. Certified stability, distinct inferential target:
   Anani et al., ICML 2026:
   https://proceedings.mlr.press/v306/anani26a.html

8. Polyhedral selective-inference foundation:
   Lee et al., Annals of Statistics 2016:
   https://doi.org/10.1214/15-AOS1371

9. Related causal-discovery selective inference:
   Guo et al., UAI 2026:
   https://proceedings.mlr.press/v337/guo26a.html

Use the same selected heads and the same total observation
budget in paired method comparisons.

NEGATIVE CONTROLS:
  A. No-op patch: identical activation replacement must
     reproduce the original output to numerical tolerance.

  B. Random-head selection: test the importance of
     adaptive ranking versus a fixed randomized choice.
     Random heads need not have zero causal effects.

  C. Covariance-diagonal ablation: remove cross-head
     dependence from the proposed inference calculation.

  D. Permuted component correspondence: disrupt
     cross-component effect dependence only in a separate
     diagnostic permutation analysis.

  E. Unadjusted same-sample intervals: illustrate
     selection inflation and coverage loss.

  F. Independent inference data: ensure the apparent
     gain does not rely on inadvertently shared samples.

  G. Strong existing inference baseline:
     If an existing valid method performs as well as the
     new method, do not claim novel empirical superiority.

METRICS (primary / secondary):

PRIMARY:
  Average width of family-wise 95% intervals for the
  identical selected set, subject to a coverage guardrail.

  Paired method difference per replication:

    d_r = width_baseline_r - width_STCCI_r

  Positive values favor STCCI.

  Select the strongest valid baseline using calibration
  draws only; freeze it before confirmation.

CO-PRIMARY SCIENTIFIC DIAGNOSTIC:
  Selection optimism:

    O_r = mean_selected(
      discovery_estimate[e] - tau[e]
    )

  Require evidence of positive, meaningful optimism.
  Separate this from the method's efficiency claim.

SECONDARY:
  Empirical simultaneous coverage.
  Per-component marginal coverage.
  False discoveries for prespecified tau[e] <= 0 nulls.
  Retained held-out intervention effects.
  Mean individual interval width.
  Selection-frequency stability.
  Interval score, reported descriptively.
  Sensitivity to selected-set size and covariance structure.

Do not treat correct answer labels as ground-truth
neural-mechanism labels.

STATISTICAL PROTOCOL
(seeds, CI method, MESOI + justification, power, TOST):

  Minimum independent sampling replications: 5.
  Calibration: 100 additional independent Monte Carlo
               replications from fixed empirical P_N.
  Confirmation: up to 1000 independent replications.

  Main unit: independently randomized selection/inference
             draw, conditional on fixed P_N.

  Use paired t-based 95% confidence intervals for the
  mean width difference, after checking heavy tails.
  Use a percentile/BCa bootstrap sensitivity analysis.

  For simultaneous coverage rates, use 95% Wilson
  binomial confidence intervals across independent draws.

  Estimate baseline width standard deviation s_B on
  calibration draws, before confirmation.

  Define the frozen primary MESOI:
    delta_width = s_B

  This defines a practically meaningful gain as one
  typical baseline replication's width variability.
  Validate its substantive interpretation before freezing.
  Do not tune delta_width using confirmatory outcomes.

  Define an analogous independently calibrated
  MESOI for selection optimism, in effect units.

  Determine achieved power from pilot variance.

  Under illustrative sigma_d = delta_width:
    five draws have roughly 40% GO power for a true
    effect of 2 * delta_width;
    five draws have roughly 29% TOST power at true zero.

  These are design-stage assumptions, NOT measured power.
  Recalculate with pilot estimates.

  The numerous CPU resampling repetitions may provide
  much greater precision for fixed-population inference.
  They do not establish generalization beyond P_N.

  TOST:
    equivalence interval [-delta_width, +delta_width].
    Two one-sided tests at alpha=0.05.
    Equivalence requires the 90% paired CI to lie
    completely inside these bounds.
    Also report the 95% confidence interval.

  Correct the two confirmatory scientific tests for
  multiplicity using Holm's procedure.
  Prespecify the primary baseline and selected k.

  Coverage guardrail:
    95% Wilson lower confidence bound on family-wise
    empirical coverage must exceed 0.925 for GO.

  The 0.925 threshold is a screening tolerance,
  not proof of 95% nominal coverage.
  Theoretical validity and detailed coverage estimates
  must still be reported.

  Use Wilson upper bound < 0.925 as evidence of
  unacceptable material undercoverage.

KILL-TEST KILL/GO/INCONCLUSIVE CRITERIA
(quantitative):

GO:
  1. Adjusted 95% CI lower bound for width improvement
     exceeds delta_width.
  2. Selection optimism is positive and exceeds its
     frozen meaningful-effect threshold.
  3. Negative controls behave as expected.
  4. Coverage guardrail is satisfied.
  5. A genuinely distinct theoretical result remains
     plausible after the novelty audit.

KILL:
  1. With adequate achieved power, the 90% paired TOST
     CI lies entirely within +/- delta_width; OR
  2. An adequately powered equivalence analysis shows
     negligible selection optimism; OR
  3. Empirical coverage upper Wilson bound < 0.925; OR
  4. Existing theory fully subsumes the proposed
     technical result; OR
  5. The required estimand or intervention is invalid
     under the verified implementation.

INCONCLUSIVE:
  1. The CI overlaps both zero and the MESOI.
  2. Calibration variance is too uncertain.
  3. Negative controls fail or numerical discrepancies
     remain unexplained.
  4. Statistical power is inadequate.
  5. Selection-conditioning approximations exhibit
     uncertain finite-sample coverage.

  In these cases:
    increase independent CPU sampling replications;
    improve numerical precision;
    increase fixed-population sample size if needed;
    recalculate compute and power;
    never redefine the hypothesis or MESOI post hoc.

COMPUTE ESTIMATE
(VRAM, T4-hours, sessions, weeks; estimates only):

Hardware:
  Kaggle T4 x2, 16 GB each, not pooled.
  No DDP.
  Use both GPUs for independent intervention batches.

Memory:
  GPT-2 fp16 weights:
    124M * 2 bytes = approximately 0.248 GB.

  At batch 8, length 128:
    residual states approx. 0.019 GB;
    unfused attention scores approx. 0.038 GB.

  Including caches, workspaces and runtime:
    estimated 2-5 GB per T4.

Initial effect cache:
  N = 1024 records
  M = 24 heads
  plus clean and corrupted forwards

  Number of forward-equivalent sequences:
    1024*(24+2) = 26,624.

  FLOP approximation:
    26,624 * 2 * 124M * 128
    = approximately 0.845 PFLOPs.

  Assume effective throughput of 0.1-0.5 TFLOP/s,
  strictly as an unmeasured planning assumption.

  Ideal compute time:
    approximately 0.47-2.35 physical T4-hours.

  With hook and framework overhead:
    estimated 2-6 physical T4-hours total.

  Sessions:
    1-2 Kaggle sessions, depending on batching.
  Weeks:
    within one nominal quota week.

  Statistical resampling:
    CPU-based after effects are cached.
    No additional transformer forward passes required.

Expanded study:
  N = 2048 existing records
  M = 144 candidate attention heads

  Approx. 299,008 forward-equivalent sequences
  including clean/corrupted evaluations.

  At length 128:
    approximately 9.49 PFLOPs before overhead.

  Expanded reservation:
    approximately 40-120 physical T4-hours,
    including hooks, confirmation and extra evaluations.

  Sessions:
    approximately 6-20 sessions depending on length
    and independent-GPU scheduling.
  Weeks:
    approximately 2-5 at a conservative
    30-hour weekly quota.

  Checkpoint before eight hours.
  Confirm the account's actual quota accounting.

PRECISION AND STABILITY:
  T4 has no bf16 support.
  Use fp16 forward where stable.
  Use fp32 logits, differences and reductions.
  Use fp64 CPU statistical inference if feasible.
  Confirm borderline effects with fp32 model forwards.
  No loss scaler needed because the model is inference-only.

FIRST IMPLEMENTATION STEPS (ordered):

  1. Create repository and freeze dependency versions.

  2. Read and record model/dataset licenses and revisions.

  3. Load MIB IOI and validate the supplied counterfactuals.

  4. Validate candidate-answer tokenization and log
     the fraction of records excluded by preprocessing.

  5. Load GPT-2 Small using TransformerBridge.

  6. Check logits against reference HF GPT-2 and verify
     required attention-head hook names and dimensions.

  7. Implement no-op patch and exact head-output
     activation patch at the last prompt token.

  8. Build deterministic effect cache with metadata:
     record ID, candidate head, effect, model revision,
     counterfactual type and processing config.

  9. Verify fp16/fp32 effect agreement on a fixed subset.

 10. Freeze the 1024-record finite empirical population.

 11. Calculate sealed reference tau vector.

 12. Implement selection and naive/honest/Bonferroni/
     max-t inference, with unit tests for coverage.

 13. Implement Gaussian top-k selection conditioning
     and selected-effect confidence intervals.

 14. Check mathematical conditioning with independent
     small-dimensional analytical examples.

 15. Run calibration draws and freeze MESOI/baselines.

 16. Run 5 smoke-test seeds and then the prespecified
     independently randomized confirmation replications.

 17. Produce interval-width, coverage, optimism, power,
     TOST and negative-control reports.

 18. Apply GO/KILL/INCONCLUSIVE rules.

 19. Only after GO and theoretical novelty clearance,
     implement greedy search-trace conditioning.

 20. Save reproducibility manifest, effect cache,
     all random seeds, config files and final reports.

REPOSITORY STRUCTURE:

  configs/
    kill_test.yaml
    full_study.yaml

  data/
    load_mib.py
    validate_counterfactuals.py
    fixed_population.py

  models/
    load_gpt2.py
    intervention_hooks.py
    effect_cache.py

  methods/
    topk_selection.py
    greedy_selection.py
    naive_intervals.py
    honest_split.py
    simultaneous_inference.py
    selection_conditioning.py

  stats/
    coverage.py
    bootstrap.py
    tost.py
    power.py
    multiplicity.py

  tests/
    test_hook_identity.py
    test_effect_cache.py
    test_selection_constraints.py
    test_conditional_interval.py
    test_split_independence.py

  reports/
    preregistration.md
    audit_novelty.md
    results.py
    reproducibility_manifest.json

KNOWN RISKS:
  A. The method may be an incremental application of
     established selective inference.

  B. Selection-event conditioning may give wider
     intervals than honest splitting.

  C. Plug-in covariance estimation may invalidate
     approximate conditional coverage.

  D. Correlated candidate effects may cause unstable
     covariance matrices or near-degenerate constraints.

  E. Conditioning on the entire search trace may discard
     too much information and lose statistical power.

  F. The finite empirical population does not represent
     arbitrary unseen linguistic templates.

  G. Strong activation effects do not prove recovery of
     the correct natural computation.

  H. Incompatible clean/corrupted tokenization may
     invalidate activation comparisons.

  I. TransformerLens API differences can silently
     change hook semantics.

  J. The initial top-k method is not, on its own, a
     sufficient novel contribution for a top-tier paper.

FINAL DIRECTIVE:
Build the smallest mathematically correct test first.
Prioritize validity, identifiability, baselines and
reproducibility over any claim of improved performance.

If the theoretical novelty gate fails, explicitly
recommend abandonment. Do not manufacture novelty by
renaming an established statistical procedure.
