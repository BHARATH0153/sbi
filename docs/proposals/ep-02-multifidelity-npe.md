# EP-02: Multi-Fidelity NPE via Fidelity Provenance Tags

Status: Discussion
Feedback: See GitHub Discussion → [EP-02 Discussion](https://github.com/sbi-dev/sbi/discussions/new?category=ideas)

## Summary

This proposal adds multi-fidelity support to the neural posterior estimation family by
tagging appended simulations with a fidelity level and letting `train()` select the
data of a single level. A density estimator is pre-trained on cheap low-fidelity
simulations and then fine-tuned on a much smaller number of expensive high-fidelity
simulations, which is the transfer-learning scheme of Goncalves et al., *Multifidelity
Simulation-based Inference for Computationally Expensive Simulators*.

## Motivation

Many simulators of interest are expensive, and an approximation of the same model is
often available at a fraction of the cost. Fitting a posterior to the expensive model
directly can require simulation budgets that are simply out of reach. Issue #1457 asks
for support for this setting.

The status quo is awkward for two reasons:

1. **Simulations of different quality are indistinguishable.** `append_simulations()`
   stores `theta` and `x` only, so once cheap and expensive runs are mixed in there is
   no way to select one of them for training.
2. **The workaround trains on everything.** Users end up concatenating the two
   fidelities, which spends the full high-fidelity budget on every epoch instead of
   spending it once, on fine-tuning.

## Goals and Non-Goals

Goals:

- Record the fidelity of each appended simulation and let `train()` restrict itself to
  one level.
- Support the pre-train / fine-tune loop, including a restricted-prior round as used by
  sequential NPE.
- Keep the existing single-fidelity call unchanged.

Non-goals:

- No simulator-owning trainer. `sbi` neural trainers never run a simulator; the
  user-driven loop (`simulate_for_sbi` → `append_simulations` → `train`) already
  expresses the schedule, and a new class would duplicate it.
- No simulation acquisition. Deciding *where* to spend the next high-fidelity budget
  (posterior-density thresholding, per-sample proposal masks) is a separate design with
  its own trade-offs, and is left for a follow-up.
- No guarantees about cost. Nothing here changes how a simulation is billed; the
  fidelity label is metadata that the caller supplies.

## Design

Two arguments carry the feature.

```python
inference = NPE_C(prior)

# Stage 1: cheap simulations, all tagged with level 0.
theta_low = prior.sample((10_000,))
inference.append_simulations(theta_low, simulate_low(theta_low), fidelity=0)
inference.train(fidelity=0)

# Stage 2: a small expensive budget, fine-tuning the same network.
theta_high = prior.sample((500,))
inference.append_simulations(theta_high, simulate_high(theta_high), fidelity=1)
inference.train(fidelity=1)
```

`append_simulations(..., fidelity=k)` stores one label per sample in
`_fidelity_roundwise`, parallel to the existing per-round storage of `theta` and `x`.
`train(fidelity=k)` sets an internal filter, and `get_simulations()` honours it, so the
dataloaders, the train/validation split, and the network initialization all operate on
the selected level only. Passing `fidelity=None` keeps the current behaviour of training
on everything appended.

Two details are worth calling out.

**The network is warm-started across levels.** With the default
`retrain_from_scratch=False`, stage 2 continues from the stage-1 weights. This is the
whole point of the transfer, and it needs no extra API.

**Round-0 fine-tuning infers the loss.** Data drawn from the prior, or from a restricted
prior covering the posterior support as in sequential NPE, is tagged as round 0 and is
fitted with the maximum-likelihood loss. `train(fidelity=...)` sets
`force_first_round_loss` automatically in that case, matching what
`tests/linearGaussian_snpe_test.py` currently spells out by hand. An explicit
`force_first_round_loss` argument still wins.

For bookkeeping, `get_fidelity_counts()` reports the number of stored simulations per
level, and `train()` records both that mapping and the level it trained on in
`inference.summary`.

### Constraints this design accepts

All levels must emit the same `x` event shape, because a single density estimator is
fit across them. Levels that differ in shape have to be reduced by a shared summary
statistic or an embedding net; appending a level with a different shape raises. Levels
must also share the `theta` space, and a coarse model that omits a parameter the fine
model resolves is expected to be padded with dummy entries.

## Alternatives Considered

**A dedicated `MFNPE` class.** This was the original suggestion on #1457. It was
dropped because it reintroduces a trainer that owns a simulator, which the current `sbi`
architecture deliberately avoids, and because it would need its own parallel
implementation of storage, dataloaders, warm-starting, and restricted priors.

**Mixing levels with per-sample weights.** Weighting levels inside one objective would
change `NeuralInference` for all methods and would need a principled weighting scheme.
Restricting the data instead reuses the existing MLE machinery and keeps the change
local to the NPE trainers.

**A separate container for fidelity-tagged simulations.** More explicit, but it would
break `get_simulations()` and every downstream consumer for no behavioural gain, since
the tag is one extra column of metadata.

## Backward Compatibility

`fidelity` defaults to `None` in both `append_simulations` and `train`, so existing code
is unaffected. Objects pickled before this change carry no fidelity bookkeeping;
`__setstate__` backfills it and treats everything stored before as level 0, so previously
saved trainers stay loadable and keep training on all their data.

Two combinations are rejected rather than silently mis-executed. `train()` with a
fidelity level that has no simulations raises instead of quietly falling back to all
data. `train(resume_training=True, fidelity=k)` raises, because resuming reuses the
train/validation indices of the previous call, which were computed over the unfiltered
dataset and would index the wrong rows.

## Testing & Validation

`tests/multifidelity_npe_test.py` covers the behaviour rather than the empirical claim:
per-sample storage and counts, that `train(fidelity=k)` sees exactly level `k` and fewer
samples than the unfiltered call, warm-starting between levels, a restricted-prior
round, `retrain_from_scratch` under a filter, unknown and invalid fidelity labels,
mismatching event shapes, and `fidelity=None` retaining single-fidelity behaviour. The
family acceptance of the new argument is checked across `NPE_A`, `NPE_B`, `NPE_C`, and
`MNPE`.

One slow test compares the multi-fidelity posterior against the analytic posterior of the
high-fidelity simulator. Its seed is pinned deliberately: across seeds the assertion is
reliable about four times in five, because short neural training on a toy problem
occasionally early-stops while the network still carries the wider low-fidelity
solution. The rest of the coverage is deterministic.

Reproducing the paper's efficiency result is left to benchmark code rather than to this
test suite.

## References

- Issue: https://github.com/sbi-dev/sbi/issues/1457
- Goncalves et al., *Multifidelity Simulation-based Inference for Computationally
  Expensive Simulators*: https://arxiv.org/abs/2502.08416
- Reference implementation: https://github.com/goncalab/multifidelity-NPE
- Process: [EP-00](https://github.com/sbi-dev/sbi/blob/main/docs/proposals/ep-00-process.md)
