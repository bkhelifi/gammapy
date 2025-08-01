# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from itertools import repeat
import numpy as np
from astropy import units as u
from astropy.table import Table
import gammapy.utils.parallel as parallel
from gammapy.datasets import Datasets
from gammapy.datasets.actors import DatasetsActor
from gammapy.datasets.flux_points import _get_reference_model
from gammapy.maps import MapAxis
from gammapy.modeling import Fit
from ..flux import FluxEstimator
from .core import FluxPoints

log = logging.getLogger(__name__)

__all__ = ["FluxPointsEstimator"]


class FluxPointsEstimator(FluxEstimator, parallel.ParallelMixin):
    """Flux points estimator.

    Estimates flux points for a given list of datasets, energies and spectral model.

    To estimate the flux point the amplitude of the reference spectral model is
    fitted within the energy range defined by the energy group. This is done for
    each group independently. The amplitude is re-normalized using the "norm" parameter,
    which specifies the deviation of the flux from the reference model in this
    energy group. See https://gamma-astro-data-formats.readthedocs.io/en/latest/spectra/binned_likelihoods/index.html
    for details.

    The method is also described in the `Fermi-LAT catalog paper <https://ui.adsabs.harvard.edu/abs/2015ApJS..218...23A>`__
    or the `H.E.S.S. Galactic Plane Survey paper <https://ui.adsabs.harvard.edu/abs/2018A%26A...612A...1H>`__

    Parameters
    ----------
    source : str or int
        For which source in the model to compute the flux points.
    n_sigma : float, optional
        Number of sigma to use for asymmetric error computation. Must be a positive value.
        Default is 1.
    n_sigma_ul : float, optional
        Number of sigma to use for upper limit computation. Must be a positive value.
        Default is 2.
    n_sigma_sensitivity : float, optional
        Sigma to use for sensitivity computation. Must be a positive value.
        Default is 5.
    selection_optional : list of str, optional
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed.
            * "errn-errp": estimate asymmetric errors on flux.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.
            * "sensitivity": estimate sensitivity for a given significance.

        Default is None so the optional steps are not executed.
    energy_edges : list of `~astropy.units.Quantity`, optional
        Edges of the flux points energy bins. The resulting bin edges won't be exactly equal to the input ones,
        but rather the closest values to the energy axis edges of the parent dataset.
        Default is [1, 10] TeV.
    fit : `Fit`, optional
        Fit instance specifying the backend and fit options. If None, the `~gammapy.modeling.Fit` instance is created
        internally. Default is None.
    reoptimize : bool, optional
        If True, the free parameters of the other models are fitted in each bin independently,
        together with the norm of the source of interest
        (but the other parameters of the source of interest are kept frozen).
        If False, only the norm of the source of interest is fitted,
        and all other parameters are frozen at their current values.
        Default is False.
    sum_over_energy_groups : bool, optional
        Whether to sum over the energy groups or fit the norm on the full energy grid. Default is None.
    n_jobs : int, optional
        Number of processes used in parallel for the computation. The number of jobs is limited to the number of
        physical CPUs. If None, defaults to `~gammapy.utils.parallel.N_JOBS_DEFAULT`.
        Default is None.
    parallel_backend : {"multiprocessing", "ray"}, optional
        Which backend to use for multiprocessing. If None, defaults to `~gammapy.utils.parallel.BACKEND_DEFAULT`.
    norm : `~gammapy.modeling.Parameter` or dict, optional
        Norm parameter used for the fit.
        Default is None and a new parameter is created automatically, with value=1, name="norm",
        scan_min=0.2, scan_max=5, and scan_n_values = 11. By default, the min and max are not set
        (consider setting them if errors or upper limits computation fails). If a dict is given,
        the entries should be a subset of `~gammapy.modeling.Parameter` arguments.

    Notes
    -----
    - For further explanation, see :ref:`estimators`.
    - In case of failure of upper limits computation (e.g. nan), see the User Guide :ref:`how_to`.

    Examples
    --------
    .. testcode::

        from astropy import units as u
        from gammapy.datasets import SpectrumDatasetOnOff
        from gammapy.estimators import FluxPointsEstimator
        from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

        path = "$GAMMAPY_DATA/joint-crab/spectra/hess/"
        dataset = SpectrumDatasetOnOff.read(path + "pha_obs23523.fits")

        pwl = PowerLawSpectralModel(index=2.7, amplitude='3e-11  cm-2 s-1 TeV-1')

        dataset.models = SkyModel(spectral_model=pwl, name="crab")

        estimator = FluxPointsEstimator(
            source="crab",
            energy_edges=[0.1, 0.3, 1, 3, 10, 30, 100] * u.TeV,
        )

        fp = estimator.run(dataset)
        print(fp)

    .. testoutput::

        FluxPoints
        ----------

          geom                   : RegionGeom
          axes                   : ['lon', 'lat', 'energy']
          shape                  : (1, 1, 6)
          quantities             : ['norm', 'norm_err', 'ts', 'npred', 'npred_excess', 'stat', 'stat_null', 'counts', 'success']
          ref. model             : pl
          n_sigma                : 1
          n_sigma_ul             : 2
          sqrt_ts_threshold_ul   : 2
          sed type init          : likelihood
    """

    tag = "FluxPointsEstimator"

    def __init__(
        self,
        energy_edges=[1, 10] * u.TeV,
        sum_over_energy_groups=False,
        n_jobs=None,
        parallel_backend=None,
        **kwargs,
    ):
        self.energy_edges = energy_edges
        self.sum_over_energy_groups = sum_over_energy_groups
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

        fit = Fit(confidence_opts={"backend": "scipy"})
        kwargs.setdefault("fit", fit)
        super().__init__(**kwargs)

    def run(self, datasets):
        """Run the flux point estimator for all energy groups.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.

        Returns
        -------
        flux_points : `FluxPoints`
            Estimated flux points.
        """
        if not isinstance(datasets, DatasetsActor):
            datasets = Datasets(datasets=datasets)

        if not datasets.energy_axes_are_aligned:
            raise ValueError("All datasets must have aligned energy axes.")

        telescopes = []
        for d in datasets:
            if d.meta_table is not None and "TELESCOP" in d.meta_table.colnames:
                telescopes.extend(list(d.meta_table["TELESCOP"].flatten()))
        if len(np.unique(telescopes)) > 1:
            raise ValueError(
                "All datasets must use the same value of the"
                " 'TELESCOP' meta keyword."
            )

        meta = {
            "n_sigma": self.n_sigma,
            "n_sigma_ul": self.n_sigma_ul,
            "sed_type_init": "likelihood",
        }

        rows = parallel.run_multiprocessing(
            self.estimate_flux_point,
            zip(
                repeat(datasets),
                self.energy_edges[:-1],
                self.energy_edges[1:],
            ),
            backend=self.parallel_backend,
            pool_kwargs=dict(processes=self.n_jobs),
            task_name="Energy bins",
        )

        table = Table(rows, meta=meta)
        model = _get_reference_model(datasets.models[self.source], self.energy_edges)
        return FluxPoints.from_table(
            table=table,
            reference_model=model.copy(),
            gti=datasets.gti,
            format="gadf-sed",
        )

    def estimate_flux_point(self, datasets, energy_min, energy_max):
        """Estimate flux point for a single energy group.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets.
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        result : dict
            Dictionary with results for the flux point.
        """
        datasets_sliced = datasets.slice_by_energy(
            energy_min=energy_min, energy_max=energy_max
        )
        if self.sum_over_energy_groups:
            datasets_sliced = datasets_sliced.__class__(
                [_.to_image(name=_.name) for _ in datasets_sliced]
            )

        if len(datasets_sliced) > 0:
            if datasets.models is not None:
                models_sliced = datasets.models._slice_by_energy(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    sum_over_energy_groups=self.sum_over_energy_groups,
                )
                datasets_sliced.models = models_sliced

            return super().run(datasets=datasets_sliced)
        else:
            log.warning(f"No dataset contribute in range {energy_min}-{energy_max}")
            model = _get_reference_model(
                datasets.models[self.source], self.energy_edges
            )
            return self._nan_result(datasets, model, energy_min, energy_max)

    def _nan_result(self, datasets, model, energy_min, energy_max):
        """NaN result."""
        energy_axis = MapAxis.from_energy_edges([energy_min, energy_max])

        with np.errstate(invalid="ignore", divide="ignore"):
            result = model.reference_fluxes(energy_axis=energy_axis)
            # convert to scalar values
            result = {key: value.item() for key, value in result.items()}

        result.update(
            {
                "norm": np.nan,
                "stat": np.nan,
                "success": False,
                "norm_err": np.nan,
                "ts": np.nan,
                "counts": np.zeros(len(datasets)),
                "npred": np.nan * np.zeros(len(datasets)),
                "npred_excess": np.nan * np.zeros(len(datasets)),
                "datasets": datasets.names,
            }
        )

        if "errn-errp" in self.selection_optional:
            result.update({"norm_errp": np.nan, "norm_errn": np.nan})

        if "ul" in self.selection_optional:
            result.update({"norm_ul": np.nan})

        if "scan" in self.selection_optional:
            norm_scan = self.norm.copy().scan_values
            result.update({"norm_scan": norm_scan, "stat_scan": np.nan * norm_scan})

        if "sensitivity" in self.selection_optional:
            result.update({"norm_sensitivity": np.nan})

        return result
