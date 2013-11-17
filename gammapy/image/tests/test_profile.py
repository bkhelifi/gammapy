# Licensed under a 3-clause BSD style license - see LICENSE.rst
from __future__ import print_function, division
from numpy.testing import assert_allclose
from astropy.tests.helper import pytest
from .. import profile

try:
    import pandas
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

@pytest.mark.skipif('not HAS_PANDAS')
def test_compute_binning():
    data = [1, 3, 2, 2, 4]
    bin_edges = profile.compute_binning(data, n_bins=3, method='equal width')
    assert_allclose(bin_edges, [1, 2, 3, 4])
    
    bin_edges = profile.compute_binning(data, n_bins=3, method='equal entries')
    # TODO: create test-cases that have been verified by hand here!
    assert_allclose(bin_edges, [1,  2,  2.66666667,  4])
    
