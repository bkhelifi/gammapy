# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from gammapy.data import GTI
from gammapy.utils.metadata import CreatorMetaData
from gammapy.version import version
from ..metadata import FluxMetaData


def test_creator():
    default = FluxMetaData(
        creation=CreatorMetaData(date="2022-01-01", creator="gammapy", origin="CTA"),
        instrument="CTAS",
        target_position=SkyCoord(83.633 * u.deg, 22.014 * u.deg, frame="icrs"),
        n_sigma=2.0,
        obs_ids=[1, 2, 3],
        dataset_names=["aa", "tt"],
    )

    assert default.creation.creator == "gammapy"
    assert default.dataset_names[1] == "tt"
    assert default.gti is None
    assert default.sed_type_init is None

    default.target_position = None
    assert np.isnan(default.target_position.ra)

    tt = Time.now()
    mgti = GTI.from_time_intervals(([tt, tt + 1 * u.s], [tt + 5 * u.s, tt + 10 * u.s]))
    default.gti = mgti

    default = FluxMetaData.from_default()
    print(default)
    assert default.instrument is None
    assert default.creation.creator == f"Gammapy {version}"
