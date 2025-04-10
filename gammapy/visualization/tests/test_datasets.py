# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
from numpy.testing import assert_allclose
import matplotlib
import matplotlib.pyplot as plt
from packaging import version
from gammapy.datasets.tests.test_map import MapDataset
from gammapy.modeling.models import (
    FoVBackgroundModel,
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import mpl_plot_check, requires_data
from gammapy.visualization import plot_npred_signal, plot_spectrum_datasets_off_regions
from gammapy.catalog import SourceCatalog3FHL


@pytest.fixture
def sky_model():
    spatial_model = GaussianSpatialModel(
        lon_0="0.2 deg", lat_0="0.1 deg", sigma="0.2 deg", frame="galactic"
    )
    spectral_model = PowerLawSpectralModel(
        index=3, amplitude="1e-11 cm-2 s-1 TeV-1", reference="1 TeV"
    )
    return SkyModel(
        spatial_model=spatial_model, spectral_model=spectral_model, name="test-model"
    )


@pytest.mark.skipif(
    version.parse(matplotlib.__version__) < version.parse("3.5"),
    reason="Requires matplotlib 3.5 or higher",
)
def test_plot_spectrum_datasets_off_regions():
    from gammapy.datasets import SpectrumDatasetOnOff
    from gammapy.maps import Map, RegionNDMap

    counts_off_1 = RegionNDMap.create("icrs;circle(0, 0.5, 0.2);circle(0.5, 0, 0.2)")

    counts_off_2 = RegionNDMap.create("icrs;circle(0.5, 0.5, 0.2);circle(0, 0, 0.2)")

    counts_off_3 = RegionNDMap.create("icrs;point(0.5, 0.5);point(0, 0)")

    m = Map.from_geom(geom=counts_off_1.geom.to_wcs_geom())
    ax = m.plot()

    dataset_1 = SpectrumDatasetOnOff(counts_off=counts_off_1)

    dataset_2 = SpectrumDatasetOnOff(counts_off=counts_off_2)

    dataset_3 = SpectrumDatasetOnOff(counts_off=counts_off_3)

    plot_spectrum_datasets_off_regions(
        ax=ax, datasets=[dataset_1, dataset_2, dataset_3]
    )

    actual = ax.patches[0].get_edgecolor()
    assert_allclose(actual, (0.121569, 0.466667, 0.705882, 1.0), rtol=1e-2)

    actual = ax.patches[2].get_edgecolor()
    assert_allclose(actual, (1.0, 0.498039, 0.054902, 1.0), rtol=1e-2)
    assert ax.lines[0].get_color() == "#2ca02c"


@requires_data()
def test_plot_regions_color_point():
    fermi_dataset = MapDataset.read(
        "$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc.fits.gz", name="fermi_dataset"
    )
    catalog = SourceCatalog3FHL()
    models = catalog.to_models().select_from_geom(fermi_dataset.geoms["geom"])

    plt.figure()
    ax = models.plot_regions(linewidth=1)
    for patch in ax.patches:
        assert_allclose(
            patch.get_edgecolor(), (0.121569, 0.466667, 0.705882, 1), rtol=1e-2
        )
    for patch in ax.lines:
        assert patch.get_color() == "C0"
    # default blue for both lines and points

    plt.figure()
    ax = models.plot_regions(linewidth=1, color="red")
    for patch in ax.patches:
        assert_allclose(patch.get_edgecolor(), (1.0, 0.0, 0.0, 1.0), rtol=1e-2)
    for patch in ax.lines:
        assert patch.get_color() == "red"
    # red because color set for both lines and points

    plt.figure()
    ax = models.plot_regions(
        linewidth=1, kwargs_point={"marker": "d", "markersize": 5, "color": "green"}
    )
    for patch in ax.patches:
        assert_allclose(
            patch.get_edgecolor(), (0.121569, 0.466667, 0.705882, 1), rtol=1e-2
        )
    for patch in ax.lines:
        assert patch.get_color() == "green"

    plt.figure()
    ax = models.plot_regions(
        linewidth=1,
        color="red",
        kwargs_point={"marker": "d", "markersize": 5, "color": "green"},
    )
    for patch in ax.patches:
        assert_allclose(patch.get_edgecolor(), (1.0, 0.0, 0.0, 1.0), rtol=1e-2)
    for patch in ax.lines:
        assert patch.get_color() == "green"

    plt.figure()
    ax = models.plot_regions(
        linewidth=1,
        color="red",
        edgecolor="green",
        kwargs_point={"marker": "d", "markersize": 5, "markeredgecolor": "green"},
    )
    for patch in ax.patches:
        assert_allclose(patch.get_edgecolor(), (1.0, 0.0, 0.0, 1.0), rtol=1e-2)
    for patch in ax.lines:
        assert patch.get_color() == "red"
    # red because color has priority over edgecolor as for the lines


@requires_data()
def test_plot_npred_signal(sky_model):
    dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")

    pwl = PowerLawSpectralModel()
    gauss = GaussianSpatialModel(
        lon_0="0.0 deg", lat_0="0.0 deg", sigma="0.5 deg", frame="galactic"
    )
    model1 = SkyModel(pwl, gauss, name="m1")

    bkg = FoVBackgroundModel(dataset_name=dataset.name)
    dataset.models = [bkg, sky_model, model1]

    with mpl_plot_check():
        plot_npred_signal(dataset)

    with mpl_plot_check():
        plot_npred_signal(dataset, model_names=[sky_model.name, model1.name])
