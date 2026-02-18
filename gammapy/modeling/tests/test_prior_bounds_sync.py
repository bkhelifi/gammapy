# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from gammapy.modeling import Parameter
from gammapy.modeling.models import (
    UniformPrior,
    LogUniformPrior,
    GaussianPrior,
    GeneralizedGaussianPrior,
)


def test_uniform_prior_auto_syncs_bounds_when_unset():
    # Test that UniformPrior automatically syncs parameter bounds when no explicit bounds are set
    p = Parameter("lon_0", value=0.5)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = UniformPrior(min=0.0, max=1.0)
    assert p.min == 0.0
    assert p.max == 1.0


def test_uniform_prior_does_not_override_existing_bounds():
    # Test that UniformPrior respects user-defined bounds and doesn't override them
    p = Parameter("lon_0", value=0.5, min=-5.0, max=5.0)
    p.prior = UniformPrior(min=0.0, max=1.0)
    assert p.min == -5.0
    assert p.max == 5.0


def test_loguniform_prior_auto_syncs_bounds_when_unset():
    # Test that LogUniformPrior automatically syncs parameter bounds when no explicit bounds are set
    p = Parameter("amplitude", value=1e-12)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = LogUniformPrior(min=1e-14, max=1e-10)
    assert p.min == 1e-14
    assert p.max == 1e-10


def test_loguniform_prior_does_not_override_existing_bounds():
    # Test that LogUniformPrior respects user-defined bounds and doesn't override them
    p = Parameter("amplitude", value=1e-12, min=1e-15, max=1e-9)
    p.prior = LogUniformPrior(min=1e-14, max=1e-10)
    assert p.min == 1e-15
    assert p.max == 1e-9


def test_gaussian_prior_does_not_set_bounds():
    # Test that GaussianPrior doesn't set bounds (it never returns inf)
    p = Parameter("index", value=2.0)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = GaussianPrior(mu=2.0, sigma=0.2)
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_generalized_gaussian_prior_does_not_set_bounds():
    # Test that GeneralizedGaussianPrior doesn't set bounds (it never returns inf)
    p = Parameter("index", value=2.0)
    assert np.isnan(p.min)
    assert np.isnan(p.max)
    p.prior = GeneralizedGaussianPrior(mu=2.0, sigma=0.2)
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_prior_modification_updates_bounds_dynamically():
    # Test that parameter bounds update automatically when prior bounds are modified
    p = Parameter("lon_0", value=0.5)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert p.min == -1.0
    assert p.max == 1.0
    p.prior.min.value = -2.0
    p.prior.max.value = 2.0
    assert p.min == -2.0
    assert p.max == 2.0


def test_setting_explicit_bounds_breaks_sync():
    # Test that setting explicit bounds stops the sync with prior bounds
    p = Parameter("lon_0", value=0.5)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert p.min == -1.0
    assert p.max == 1.0
    p.min = -5.0
    p.max = 5.0
    assert p.min == -5.0
    assert p.max == 5.0
    p.prior.min.value = -10.0
    p.prior.max.value = 10.0
    assert p.min == -5.0
    assert p.max == 5.0


def test_clearing_prior_restores_nan_bounds():
    # Test that clearing the prior restores nan bounds if they were synced
    p = Parameter("lon_0", value=0.5)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert p.min == -1.0
    assert p.max == 1.0
    p.prior = None
    assert np.isnan(p.min)
    assert np.isnan(p.max)


def test_partial_bounds_sync():
    # Test that only unset bounds sync with prior (partial sync scenario)
    p = Parameter("lon_0", value=0.5, min=-5.0)
    p.prior = UniformPrior(min=-1.0, max=1.0)
    assert p.min == -5.0
    assert p.max == 1.0
    p.max = 5.0
    assert p.min == -5.0
    assert p.max == 5.0


def test_factor_min_max_use_synced_bounds():
    # Test that factor_min and factor_max correctly use the synced bounds
    p = Parameter("amplitude", value=1e-12, scale=1e-12)
    p.prior = UniformPrior(min=0.0, max=1e-10)
    assert_allclose(p.factor_min, 0.0)
    assert_allclose(p.factor_max, 1e-10 / 1e-12)
