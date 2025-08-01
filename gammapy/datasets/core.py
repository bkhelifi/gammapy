# Licensed under a 3-clause BSD style license - see LICENSE.rst
import abc
import collections.abc
import copy
import html
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from gammapy.data import GTI
from gammapy.modeling.models import DatasetModels, Models
from gammapy.utils.scripts import make_name, make_path, read_yaml, to_yaml, write_yaml
from gammapy.stats import FIT_STATISTICS_REGISTRY

log = logging.getLogger(__name__)


__all__ = ["Dataset", "Datasets"]


class Dataset(abc.ABC):
    """Dataset abstract base class.
    For now, see existing examples of type of datasets:

    - `gammapy.datasets.MapDataset`
    - `gammapy.datasets.SpectrumDataset`
    - `gammapy.datasets.FluxPointsDataset`

    For more information see :ref:`datasets`.
    """

    # TODO: add tutorial how to create your own dataset types.
    _residuals_labels = {
        "diff": "data - model",
        "diff/model": "(data - model) / model",
        "diff/sqrt(model)": "(data - model) / sqrt(model)",
    }

    @property
    def stat_type(self):
        """The Fit Statistic class used."""
        return self._stat_type

    @stat_type.setter
    def stat_type(self, stat_type):
        """Set the Fit Statistic."""
        self._fit_statistic = FIT_STATISTICS_REGISTRY[stat_type]
        self._stat_type = stat_type

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    @property
    @abc.abstractmethod
    def tag(self):
        pass

    @property
    def name(self):
        return self._name

    def to_dict(self):
        """Convert to dict for YAML serialization."""
        name = self.name.replace(" ", "_")
        filename = f"{name}.fits"
        return {"name": self.name, "type": self.tag, "filename": filename}

    @property
    def mask(self):
        """Combined fit and safe mask."""
        if self.mask_safe is not None and self.mask_fit is not None:
            return self.mask_safe * self.mask_fit
        elif self.mask_fit is not None:
            return self.mask_fit
        elif self.mask_safe is not None:
            return self.mask_safe

    def stat_sum(self):
        """Total statistic given the current model parameters and priors."""
        return self._fit_statistic.stat_sum_dataset(self)

    def _stat_sum_likelihood(self):
        """Total statistic given the current model parameters without the priors."""
        return self._fit_statistic.stat_sum_dataset(self)

    def stat_array(self):
        """Statistic array, one value per data point."""
        return self._fit_statistic.stat_array_dataset(self)

    def copy(self, name=None):
        """A deep copy.

        Parameters
        ----------
        name : str, optional
            Name of the copied dataset. Default is None.

        Returns
        -------
        dataset : `Dataset`
            Copied datasets.
        """
        new = copy.deepcopy(self)
        name = make_name(name)
        new._name = name
        # TODO: check the model behaviour?
        new.models = None
        return new

    @staticmethod
    def _compute_residuals(data, model, method="diff"):
        with np.errstate(invalid="ignore"):
            if method == "diff":
                residuals = data - model
            elif method == "diff/model":
                residuals = (data - model) / model
            elif method == "diff/sqrt(model)":
                residuals = (data - model) / np.sqrt(model)
            else:
                raise AttributeError(
                    f"Invalid method: {method!r} for computing residuals"
                )
        return residuals


class Datasets(collections.abc.MutableSequence):
    """Container class that holds a list of datasets.

    Parameters
    ----------
    datasets : `Dataset` or list of `Dataset`
        Datasets.
    """

    def __init__(self, datasets=None):
        if datasets is None:
            datasets = []

        if isinstance(datasets, Datasets):
            datasets = datasets._datasets
        elif isinstance(datasets, Dataset):
            datasets = [datasets]
        elif not isinstance(datasets, list):
            raise TypeError(f"Invalid type: {datasets!r}")

        unique_names = []
        for dataset in datasets:
            if dataset.name in unique_names:
                raise (ValueError("Dataset names must be unique"))
            unique_names.append(dataset.name)

        self._datasets = datasets
        self._covariance = None

    @property
    def parameters(self):
        """Unique parameters (`~gammapy.modeling.Parameters`).

        Duplicate parameter objects have been removed.
        The order of the unique parameters remains.
        """
        return self.models.parameters.unique_parameters

    @property
    def models(self):
        """Unique models (`~gammapy.modeling.Models`).

        Duplicate model objects have been removed.
        The order of the unique models remains.
        """
        models = {}

        for dataset in self:
            if dataset.models is not None:
                for model in dataset.models:
                    models[model] = model
        models = DatasetModels(list(models.keys()))

        if self._covariance and self._covariance.parameters == models.parameters:
            return DatasetModels(models, covariance_data=self._covariance.data)
        else:
            return models

    @models.setter
    def models(self, models):
        """Unique models (`~gammapy.modeling.Models`).

        Duplicate model objects have been removed.
        The order of the unique models remains.
        """
        if models:
            self._covariance = DatasetModels(models).covariance
        for dataset in self:
            dataset.models = models

    @property
    def names(self):
        return [d.name for d in self._datasets]

    @property
    def is_all_same_type(self):
        """Whether all contained datasets are of the same type."""
        return len(set(_.__class__ for _ in self)) == 1

    @property
    def is_all_same_shape(self):
        """Whether all contained datasets have the same data shape."""
        return len(set(_.data_shape for _ in self)) == 1

    @property
    def is_all_same_energy_shape(self):
        """Whether all contained datasets have the same data shape."""
        return len(set(_.data_shape[0] for _ in self)) == 1

    @property
    def energy_axes_are_aligned(self):
        """Whether all contained datasets have aligned energy axis."""
        axes = [d.counts.geom.axes["energy"] for d in self]
        return np.all([axes[0].is_aligned(ax) for ax in axes])

    @property
    def contributes_to_stat(self):
        """Stat contributions.

        Returns
        -------
        contributions : `~numpy.array`
            Array indicating which dataset contributes to the likelihood.
        """
        contributions = []

        for dataset in self:
            if dataset.mask is not None:
                value = np.any(dataset.mask)
            else:
                value = True
            contributions.append(value)
        return np.array(contributions)

    def stat_sum(self):
        """Compute joint statistic function value."""
        prior_stat_sum = 0.0
        if self.models is not None:
            prior_stat_sum = self.models.parameters.prior_stat_sum()

        stat_sum = 0.0
        for dataset in self:
            stat_sum += dataset.stat_sum()

        return stat_sum + prior_stat_sum

    def _stat_sum_likelihood(self):
        """Total statistic given the current model parameters without the priors."""
        stat_sum = 0
        for dataset in self:
            stat_sum += dataset._stat_sum_likelihood()
        return stat_sum

    def select_time(self, time_min, time_max, atol="1e-6 s"):
        """Select datasets in a given time interval.

        Parameters
        ----------
        time_min, time_max : `~astropy.time.Time`
            Time interval.
        atol : `~astropy.units.Quantity`
            Tolerance value for time comparison with different scale. Default 1e-6 sec.

        Returns
        -------
        datasets : `Datasets`
            Datasets in the given time interval.

        """
        atol = u.Quantity(atol)

        datasets = []

        for dataset in self:
            t_start = dataset.gti.time_start[0]
            t_stop = dataset.gti.time_stop[-1]

            if t_start >= (time_min - atol) and t_stop <= (time_max + atol):
                datasets.append(dataset)

        return self.__class__(datasets)

    def slice_by_energy(self, energy_min, energy_max):
        """Select and slice datasets in energy range.

        The method keeps the current dataset names. Datasets that do not
        contribute to the selected energy range are dismissed.

        Parameters
        ----------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy bounds to compute the flux point for.

        Returns
        -------
        datasets : Datasets
            Datasets.

        """
        datasets = []

        for dataset in self:
            try:
                dataset_sliced = dataset.slice_by_energy(
                    energy_min=energy_min,
                    energy_max=energy_max,
                    name=dataset.name,
                )
            except ValueError:
                log.info(
                    f"Dataset {dataset.name} does not contribute in the energy range"
                )
                continue

            datasets.append(dataset_sliced)

        return self.__class__(datasets=datasets)

    def to_spectrum_datasets(self, region):
        """Extract spectrum datasets for the given region.

        To get more detailed information, see the corresponding function associated to each dataset type:
        `~gammapy.datasets.MapDataset.to_spectrum_dataset` or `~gammapy.datasets.MapDatasetOnOff.to_spectrum_dataset`.

        Parameters
        ----------
        region : `~regions.SkyRegion`
            Region definition.

        Returns
        -------
        datasets : `Datasets`
            List of `~gammapy.datasets.SpectrumDataset`.
        """
        datasets = Datasets()

        for dataset in self:
            spectrum_dataset = dataset.to_spectrum_dataset(
                on_region=region, name=dataset.name
            )
            datasets.append(spectrum_dataset)

        return datasets

    def _to_asimov_datasets(self):
        """Create Asimov datasets from the current models."""
        return Datasets([d._to_asimov_dataset() for d in self])

    @property
    # TODO: make this a method to support different methods?
    def energy_ranges(self):
        """Get global energy range of datasets.

        The energy range is derived as the minimum / maximum of the energy
        ranges of all datasets.

        Returns
        -------
        energy_min, energy_max : `~astropy.units.Quantity`
            Energy range.
        """

        energy_mins, energy_maxs = [], []

        for dataset in self:
            energy_axis = dataset.counts.geom.axes["energy"]
            energy_mins.append(energy_axis.edges[0])
            energy_maxs.append(energy_axis.edges[-1])

        return u.Quantity(energy_mins), u.Quantity(energy_maxs)

    def __str__(self):
        str_ = self.__class__.__name__ + "\n"
        str_ += "--------\n\n"

        for idx, dataset in enumerate(self):
            str_ += f"Dataset {idx}: \n\n"
            str_ += f"\tType       : {dataset.tag}\n"
            str_ += f"\tName       : {dataset.name}\n"
            try:
                instrument = set(dataset.meta_table["TELESCOP"]).pop()
            except (KeyError, TypeError):
                instrument = ""
            str_ += f"\tInstrument : {instrument}\n"
            if dataset.models:
                names = dataset.models.names
            else:
                names = ""
            str_ += f"\tModels     : {names}\n\n"

        return str_.expandtabs(tabsize=2)

    def _repr_html_(self):
        try:
            return self.to_html()
        except AttributeError:
            return f"<pre>{html.escape(str(self))}</pre>"

    def copy(self):
        """A deep copy."""
        return copy.deepcopy(self)

    @classmethod
    def read(cls, filename, filename_models=None, lazy=True, cache=True, checksum=True):
        """De-serialize datasets from YAML and FITS files.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            File path or name of datasets yaml file.
        filename_models : str or `~pathlib.Path`, optional
            File path or name of models yaml file. Default is None.
        lazy : bool
            Whether to lazy load data into memory. Default is True.
        cache : bool
            Whether to cache the data after loading. Default is True.
        checksum : bool
            Whether to perform checksum verification. Default is False.

        Returns
        -------
        dataset : `gammapy.datasets.Datasets`
            Datasets.
        """
        from . import DATASET_REGISTRY

        filename = make_path(filename)
        data_list = read_yaml(filename, checksum=checksum)

        datasets = []
        for data in data_list["datasets"]:
            path = filename.parent

            if (path / data["filename"]).exists():
                data["filename"] = str(make_path(path / data["filename"]))

            dataset_cls = DATASET_REGISTRY.get_cls(data["type"])
            dataset = dataset_cls.from_dict(data, lazy=lazy, cache=cache)
            datasets.append(dataset)

        datasets = cls(datasets)

        if filename_models:
            datasets.models = Models.read(filename_models, checksum=checksum)

        return datasets

    def write(
        self,
        filename,
        filename_models=None,
        overwrite=False,
        write_covariance=True,
        checksum=True,
    ):
        """Serialize datasets to YAML and FITS files.

        Parameters
        ----------
        filename : str or `~pathlib.Path`
            File path or name of datasets yaml file.
        filename_models : str or `~pathlib.Path`, optional
            File path or name of models yaml file. Default is None.
        overwrite : bool, optional
            Overwrite existing file. Default is False.
        write_covariance : bool
            save covariance or not. Default is True.
        checksum : bool
            When True adds both DATASUM and CHECKSUM cards to the headers written to the FITS files.
            Default is True.
        """
        path = make_path(filename)

        data = {"datasets": []}

        for dataset in self._datasets:
            d = dataset.to_dict()
            filename = d["filename"]
            dataset.write(
                path.parent / filename, overwrite=overwrite, checksum=checksum
            )
            data["datasets"].append(d)

        if path.exists() and not overwrite:
            raise IOError(f"File exists already: {path}")
        yaml_str = to_yaml(data)
        write_yaml(yaml_str, path, checksum=checksum, overwrite=overwrite)

        if filename_models:
            self.models.write(
                filename_models,
                overwrite=overwrite,
                write_covariance=write_covariance,
                checksum=checksum,
            )

    def stack_reduce(self, name=None, nan_to_num=True):
        """Reduce the Datasets to a unique Dataset by stacking them together.

        This works only if all datasets are of the same type and with aligned geometries, and if a proper
        in-place stack method exists for the Dataset type.

        For details, see :ref:`stack`.

        Parameters
        ----------
        name : str, optional
            Name of the stacked dataset. Default is None.
        nan_to_num : bool
            Non-finite values are replaced by zero if True. Default is True.

        Returns
        -------
        dataset : `~gammapy.datasets.Dataset`
            The stacked dataset.
        """
        if not self.is_all_same_type:
            raise ValueError(
                "Stacking impossible: all Datasets contained are not of a unique type."
            )

        stacked = self[0].to_masked(name=name, nan_to_num=nan_to_num)

        for dataset in self[1:]:
            stacked.stack(dataset, nan_to_num=nan_to_num)

        return stacked

    def info_table(self, cumulative=False):
        """Get info table for datasets.

        Parameters
        ----------
        cumulative : bool
            Cumulate information across all datasets. If True, all model-dependent
            information will be lost. Default is False.

        Returns
        -------
        info_table : `~astropy.table.Table`
            Info table.
        """
        if not self.is_all_same_type:
            raise ValueError("Info table not supported for mixed dataset type.")

        rows = []

        if cumulative:
            name = "stacked"
            stacked = self[0].to_masked(name=name)
            rows.append(stacked.info_dict())
            for dataset in self[1:]:
                stacked.stack(dataset)
                rows.append(stacked.info_dict())
        else:
            for dataset in self:
                rows.append(dataset.info_dict())

        return Table(rows)

    # TODO: merge with meta table?
    @property
    def gti(self):
        """GTI table."""
        time_intervals = []

        for dataset in self:
            if dataset.gti is not None and len(dataset.gti.table) > 0:
                interval = (dataset.gti.time_start[0], dataset.gti.time_stop[-1])
                time_intervals.append(interval)

        if len(time_intervals) == 0:
            return None

        return GTI.from_time_intervals(time_intervals)

    @property
    def meta_table(self):
        """Meta table."""
        tables = [d.meta_table for d in self]

        if np.all([table is None for table in tables]):
            meta_table = Table()
        else:
            meta_table = vstack(tables).copy()

        meta_table.add_column([d.tag for d in self], index=0, name="TYPE")
        meta_table.add_column(self.names, index=0, name="NAME")
        return meta_table

    def __getitem__(self, key):
        return self._datasets[self.index(key)]

    def __delitem__(self, key):
        del self._datasets[self.index(key)]

    def __setitem__(self, key, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets[self.index(key)] = dataset
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def insert(self, idx, dataset):
        if isinstance(dataset, Dataset):
            if dataset.name in self.names:
                raise (ValueError("Dataset names must be unique"))
            self._datasets.insert(idx, dataset)
        else:
            raise TypeError(f"Invalid type: {type(dataset)!r}")

    def index(self, key):
        if isinstance(key, (int, slice)):
            return key
        elif isinstance(key, str):
            return self.names.index(key)
        elif isinstance(key, Dataset):
            return self._datasets.index(key)
        else:
            raise TypeError(f"Invalid type: {type(key)!r}")

    def __len__(self):
        return len(self._datasets)
