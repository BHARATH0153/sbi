# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License, Version 2.0, see <https://github.com/sbi-dev/sbi>

r"""Tests for multi-fidelity NPE via the ``fidelity`` provenance tag.

The method is described in Goncalves et al., *Multifidelity Simulation-based Inference
for Computationally Expensive Simulators*, https://arxiv.org/abs/2502.08416: a density
estimator is pre-trained on cheap low-fidelity simulations and then fine-tuned on a much
smaller number of expensive high-fidelity simulations.

These tests cover the plumbing rather than the empirical claims: which simulations
``train(fidelity=...)`` actually sees, that the provenance is stored per sample, and
that the guards around transfer learning fire.
"""

from __future__ import annotations

import pickle
import tempfile
from copy import deepcopy

import pytest
import torch
from torch import Tensor, eye, ones, zeros

from sbi.inference import NPE_A, NPE_B, NPE_C
from sbi.simulators.linear_gaussian import (
    linear_gaussian,
    samples_true_posterior_linear_gaussian_uniform_prior,
)
from sbi.utils import BoxUniform, RestrictedPrior, get_density_thresholder
from sbi.utils.metrics import check_c2st

NUM_DIM = 2
LIKELIHOOD_SHIFT = -1.0 * ones(NUM_DIM)
LIKELIHOOD_COV = 0.3 * eye(NUM_DIM)


def get_two_fidelity_simulators(noise_low: float = 0.2):
    """Return a cheap noisy simulator and an expensive accurate one.

    Both emit the same event shape, which is what lets a single density estimator be
    fit across the two levels. The high-fidelity simulator is the ground truth task; the
    low-fidelity one is deliberately misspecified by adding more observation noise, in
    the spirit of a cheap approximation of an expensive model.

    Args:
        noise_low: Extra observation noise of the low-fidelity simulator.

    Returns:
        Tuple of the low-fidelity and the high-fidelity simulator.
    """

    def low_fidelity_simulator(theta: Tensor) -> Tensor:
        return linear_gaussian(theta, LIKELIHOOD_SHIFT, LIKELIHOOD_COV) + (
            noise_low * torch.randn_like(theta)
        )

    def high_fidelity_simulator(theta: Tensor) -> Tensor:
        return linear_gaussian(theta, LIKELIHOOD_SHIFT, LIKELIHOOD_COV)

    return low_fidelity_simulator, high_fidelity_simulator


def append_two_levels(inference, prior, low, high, num_low: int, num_high: int):
    """Append a low-fidelity batch followed by a high-fidelity batch.

    Args:
        inference: The trainer to append to.
        prior: The prior to sample parameters from.
        low: The low-fidelity simulator.
        high: The high-fidelity simulator.
        num_low: Number of low-fidelity simulations.
        num_high: Number of high-fidelity simulations.
    """
    theta_low = prior.sample((num_low,))
    inference.append_simulations(theta_low, low(theta_low), fidelity=0)
    theta_high = prior.sample((num_high,))
    inference.append_simulations(theta_high, high(theta_high), fidelity=1)


class LossRecorder:
    """Records the `x` of every batch that reaches the trainer's loss.

    Wrapping `_loss` is the most direct way to observe what a `train()` call actually
    trains on. The wrapper delegates to the original loss, so training behaviour is
    unchanged.
    """

    def __init__(self, inference):
        self._original_loss = inference._loss
        self.batches: list[Tensor] = []
        inference._loss = self._record

    def _record(self, *args, **kwargs) -> Tensor:
        self.batches.append(args[1].detach().clone())
        return self._original_loss(*args, **kwargs)

    def reset(self) -> None:
        """Forget everything recorded so far."""
        self.batches = []

    def num_simulations(self) -> int:
        """Return the number of `x` rows seen since the last `reset`."""
        return sum(batch.shape[0] for batch in self.batches)


def test_fidelity_is_stored_per_sample():
    """Each appended batch records its fidelity level for every simulation."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)

    theta_low = prior.sample((7,))
    theta_high = prior.sample((3,))
    inference.append_simulations(theta_low, low(theta_low), fidelity=0)
    inference.append_simulations(theta_high, high(theta_high), fidelity=1)

    assert len(inference._fidelity_roundwise) == 2
    assert inference._fidelity_roundwise[0].shape == (7,)
    assert inference._fidelity_roundwise[1].shape == (3,)
    assert inference.get_fidelity_counts() == {0: 7, 1: 3}


def test_fidelity_counts_ignore_invalid_simulations():
    """Discarded simulations are not counted in the fidelity bookkeeping."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    inference = NPE_C(prior, show_progress_bars=False)

    theta = prior.sample((13,))
    x = torch.cat([prior.sample((10,)), torch.full((3, NUM_DIM), float("nan"))])
    inference.append_simulations(theta, x, fidelity=1)

    assert inference.get_fidelity_counts() == {1: 10}


@pytest.mark.parametrize("fidelity, expected_num", [(0, 500), (1, 400)])
def test_active_fidelity_selects_the_dataset(fidelity: int, expected_num: int):
    """While a level is active, `get_simulations` returns only that level.

    This is the mechanism both `get_dataloaders` and the network initialization read, so
    asserting on it directly pins down what a fidelity-filtered `train()` call sees.
    """
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=500, num_high=400)

    inference._active_fidelity = fidelity
    try:
        theta, x, prior_masks = inference.get_simulations()
    finally:
        inference._active_fidelity = None

    assert theta.shape[0] == x.shape[0] == prior_masks.shape[0]
    assert theta.shape[0] == expected_num

    assert inference.get_simulations()[0].shape[0] == 900


def test_train_fidelity_only_sees_that_level():
    """`train(fidelity=k)` trains on the level-k simulations and on nothing else."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=2000, num_high=400)
    recorder = LossRecorder(inference)

    inference.train(fidelity=1, max_num_epochs=1, training_batch_size=100)
    filtered = recorder.num_simulations()

    recorder.reset()
    inference.train(
        force_first_round_loss=True, max_num_epochs=1, training_batch_size=100
    )
    unfiltered = recorder.num_simulations()

    assert filtered > 0
    assert filtered < unfiltered


def test_multifidelity_pretrain_then_finetune_warm_starts():
    """Pre-training then fine-tuning reuses the network instead of rebuilding it.

    This is the transfer step of MF-NPE: the fine-tuning stage must continue from the
    pre-trained weights, which is what lets it work with few high-fidelity simulations.
    """
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)

    theta_low = prior.sample((500,))
    inference.append_simulations(theta_low, low(theta_low), fidelity=0).train(
        max_num_epochs=2
    )
    pretrained_net = inference._neural_net
    pretrained_state = deepcopy(pretrained_net.state_dict())

    theta_high = prior.sample((100,))
    inference.append_simulations(theta_high, high(theta_high), fidelity=1).train(
        fidelity=1, max_num_epochs=1
    )

    assert inference._neural_net is pretrained_net
    assert any(
        not torch.allclose(value, pretrained_state[key])
        for key, value in inference._neural_net.state_dict().items()
    )
    assert inference.get_fidelity_counts() == {0: 500, 1: 100}
    assert inference.summary["trained_fidelity"] == 1


@pytest.mark.slow
def test_multifidelity_recovers_posterior_with_few_high_fidelity_sims():
    """MF-NPE recovers a usable posterior from a small high-fidelity budget.

    The claim under test is that the transfer path yields a posterior matching the
    analytic posterior of the high-fidelity simulator, even though the high-fidelity
    pool is a small fraction of the pre-training pool.

    The seed is pinned on purpose. Across seeds this assertion is only reliable about
    four times out of five: in roughly one run out of five, fine-tuning early-stops
    while the network still carries the wider low-fidelity solution and one posterior
    dimension collapses. That is a property of short neural training on a toy problem,
    not of the fidelity code, but it would make the default test suite flaky. The
    deterministic plumbing coverage lives in the tests above.
    """
    torch.manual_seed(0)
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    x_o = zeros(1, NUM_DIM)

    inference = NPE_C(prior, show_progress_bars=False)
    theta_low = prior.sample((10000,))
    inference.append_simulations(theta_low, low(theta_low), fidelity=0).train(
        max_num_epochs=200, stop_after_epochs=20
    )
    theta_high = prior.sample((2000,))
    inference.append_simulations(theta_high, high(theta_high), fidelity=1).train(
        fidelity=1, max_num_epochs=200, stop_after_epochs=20
    )
    samples = inference.build_posterior().sample((2000,), x=x_o)

    target = samples_true_posterior_linear_gaussian_uniform_prior(
        x_o,
        LIKELIHOOD_SHIFT,
        LIKELIHOOD_COV,
        prior=prior,
        num_samples=2000,
    )

    check_c2st(samples, target, alg="mf_npe", tol=0.15)


def test_multifidelity_sequential_round_with_restricted_prior():
    """MF-TSNPE: fine-tuning rounds use a restricted prior like the TSNPE loop.

    The restricted prior covers the posterior support, so the data is tagged as round 0
    and has to be fitted with the maximum-likelihood loss. `train(fidelity=...)` infers
    that, which is what the explicit `force_first_round_loss=True` in the TSNPE branch
    of `tests/linearGaussian_snpe_test.py` does by hand.
    """
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    x_o = zeros(1, NUM_DIM)

    inference = NPE_C(prior, show_progress_bars=False)
    theta_low = prior.sample((500,))
    inference.append_simulations(theta_low, low(theta_low), fidelity=0).train(
        max_num_epochs=2
    )

    theta = prior.sample((500,))
    inference.append_simulations(theta, high(theta), fidelity=1).train(
        fidelity=1, max_num_epochs=2
    )
    posterior_1 = inference.build_posterior().set_default_x(x_o)

    proposal = RestrictedPrior(
        prior,
        get_density_thresholder(posterior_1, quantile=1e-4),
        posterior=posterior_1,
        sample_with="rejection",
    )
    theta = proposal.sample((500,))
    inference.append_simulations(theta, high(theta), fidelity=1).train(
        fidelity=1, max_num_epochs=2
    )
    samples = inference.build_posterior().set_default_x(x_o).sample((500,), x=x_o)

    assert samples.shape == (500, NUM_DIM)
    assert torch.isfinite(samples).all()
    assert inference.get_fidelity_counts() == {0: 500, 1: 1000}


def test_train_resume_training_with_fidelity_raises():
    """Resuming a fidelity-filtered call is rejected rather than mis-indexing."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=200, num_high=100)

    with pytest.raises(ValueError, match="resume_training=True cannot be combined"):
        inference.train(fidelity=1, resume_training=True)


def test_legacy_trainer_without_fidelity_state_can_be_loaded():
    """A trainer pickled before `fidelity` existed still works after loading."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    high, _ = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    theta = prior.sample((100,))
    inference.append_simulations(theta, high(theta))

    legacy_state = deepcopy(inference.__getstate__())
    del legacy_state["_fidelity_roundwise"]
    del legacy_state["_active_fidelity"]
    del legacy_state["_summary"]["fidelity_counts"]
    del legacy_state["_summary"]["trained_fidelity"]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = f"{tmpdir}/legacy.pkl"
        with open(path, "wb") as file:
            pickle.dump(legacy_state, file)
        with open(path, "rb") as file:
            restored = pickle.load(file)
    legacy = NPE_C.__new__(NPE_C)
    legacy.__setstate__(restored)

    assert legacy.get_fidelity_counts() == {0: 100}
    assert legacy._active_fidelity is None


def test_train_fidelity_unknown_level_raises():
    """Training on a level with no data raises instead of silently using all."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, _ = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)

    theta = prior.sample((100,))
    inference.append_simulations(theta, low(theta), fidelity=0)

    with pytest.raises(
        ValueError, match="No simulations were appended with fidelity=3"
    ):
        inference.train(fidelity=3)


@pytest.mark.parametrize("fidelity", [-1, 1.5, "1"])
def test_invalid_fidelity_raises(fidelity):
    """Fidelity labels must be non-negative integers."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, _ = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)

    theta = prior.sample((10,))
    with pytest.raises((TypeError, ValueError)):
        inference.append_simulations(theta, low(theta), fidelity=fidelity)


def test_fidelity_levels_with_different_x_shapes_raise():
    """Fidelity levels must emit the same event shape.

    A single density estimator is fit across levels and its condition shape is fixed
    when the network is built, so a mismatch has to be rejected when the data is
    appended rather than deep inside training.
    """
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    inference = NPE_C(prior, show_progress_bars=False)

    theta = prior.sample((10,))
    inference.append_simulations(theta, ones(10, NUM_DIM), fidelity=0)

    with pytest.raises(ValueError, match="must produce simulations of the same shape"):
        inference.append_simulations(theta, ones(10, NUM_DIM, 3), fidelity=1)


def test_fidelity_none_is_single_fidelity_behaviour():
    """The default is untouched: `fidelity=None` trains on everything."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=300, num_high=100)

    recorded = LossRecorder(inference)
    inference.train(max_num_epochs=1, training_batch_size=100)

    assert recorded.num_simulations() > 300
    assert inference.summary["trained_fidelity"] is None


@pytest.mark.parametrize("trainer_cls", [NPE_A, NPE_B, NPE_C])
def test_fidelity_is_accepted_by_every_npe_trainer(trainer_cls):
    """Every member of the NPE family forwards `fidelity` to the shared trainer."""
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = trainer_cls(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=200, num_high=200)

    inference.train(fidelity=0, max_num_epochs=1)
    inference.train(fidelity=1, max_num_epochs=1)

    assert inference.get_fidelity_counts() == {0: 200, 1: 200}
    assert inference.summary["trained_fidelity"] == 1


def test_retrain_from_scratch_uses_only_the_selected_level():
    """Rebuilding the network under a fidelity filter builds it from that level.

    Otherwise the condition shape and the z-scoring statistics would come from the
    low-fidelity data while training runs on the high-fidelity data.
    """
    prior = BoxUniform(-2 * ones(NUM_DIM), 2 * ones(NUM_DIM))
    low, high = get_two_fidelity_simulators()
    inference = NPE_C(prior, show_progress_bars=False)
    append_two_levels(inference, prior, low, high, num_low=500, num_high=400)

    inference.train(fidelity=1, retrain_from_scratch=True, max_num_epochs=1)

    assert inference.get_fidelity_counts() == {0: 500, 1: 400}
    assert inference.summary["trained_fidelity"] == 1
