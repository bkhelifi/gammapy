# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Covariance class."""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from gammapy.utils.parallel import is_ray_initialized
from .parameter import Parameters

__all__ = ["Covariance"]


class Covariance:
    """Parameter covariance class.

    Parameters
    ----------
    parameters : `~gammapy.modeling.Parameters`
        Parameter list.
    data : `~numpy.ndarray`
        Covariance data array.

    """

    def __init__(self, parameters, data=None):
        self.parameters = parameters
        if data is None:
            data = np.diag([p.error**2 for p in self.parameters])

        self._data = np.asanyarray(data, dtype=float)

    @property
    def shape(self):
        """Covariance shape."""
        npars = len(self.parameters)
        return npars, npars

    @property
    def data(self):
        """Covariance data as a `~numpy.ndarray`."""
        return self._data

    @data.setter
    def data(self, value):
        value = np.asanyarray(value)

        npars = len(self.parameters)
        shape = (npars, npars)
        if value.shape != shape:
            raise ValueError(
                f"Invalid covariance shape: {value.shape}, expected {shape}"
            )

        self._data = value

    @staticmethod
    def _expand_factor_matrix(matrix, parameters):
        """Expand covariance matrix with zeros for frozen parameters."""
        npars = len(parameters)
        matrix_expanded = np.zeros((npars, npars))
        mask_frozen = [par.frozen for par in parameters]
        pars_index = [np.where(np.array(parameters) == p)[0][0] for p in parameters]
        mask_duplicate = [pars_idx != idx for idx, pars_idx in enumerate(pars_index)]
        mask = np.array(mask_frozen) | np.array(mask_duplicate)
        free_parameters = ~(mask | mask[:, np.newaxis])
        matrix_expanded[free_parameters] = matrix.ravel()
        return matrix_expanded

    @classmethod
    def from_factor_matrix(cls, parameters, matrix):
        """Set covariance from factor covariance matrix.

        Used in the optimizer interface.
        """

        npars = len(parameters)

        if npars > 0:
            if not matrix.shape == (npars, npars):
                matrix = cls._expand_factor_matrix(matrix, parameters)

            df = np.diag(
                [p._inverse_transform_derivative(p.factor) for p in parameters]
            )
            data = df.T @ matrix @ df
        else:
            data = None

        return cls(parameters, data=data)

    @classmethod
    def from_stack(cls, covar_list):
        """Stack sub-covariance matrices from list.

        Parameters
        ----------
        covar_list : list of `Covariance`
            List of sub-covariances.

        Returns
        -------
        covar : `Covariance`
            Stacked covariance.
        """
        parameters = Parameters.from_stack([_.parameters for _ in covar_list])

        covar = cls(parameters)

        for subcovar in covar_list:
            covar.set_subcovariance(subcovar)

        return covar

    def get_subcovariance(self, parameters):
        """Get sub-covariance matrix.

        Parameters
        ----------
        parameters : `Parameters`
            Sub list of parameters.

        Returns
        -------
        covariance : `~numpy.ndarray`
            Sub-covariance.
        """
        idx = [self.parameters.index(par) for par in parameters]
        data = self._data[np.ix_(idx, idx)]
        return self.__class__(parameters=parameters, data=data)

    def set_subcovariance(self, covar):
        """Set sub-covariance matrix.

        Parameters
        ----------
        covar : `Covariance`
            Sub-covariance.
        """
        if is_ray_initialized():
            # This copy is required to make the covariance setting work with ray
            self._data = self._data.copy()

        idx = [self.parameters.index(par) for par in covar.parameters]

        if not np.allclose(self.data[np.ix_(idx, idx)], covar.data):
            self.data[idx, :] = 0
            self.data[:, idx] = 0

        self._data[np.ix_(idx, idx)] = covar.data

    def plot_correlation(self, figsize=None, **kwargs):
        """Plot correlation matrix.

        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is None, which takes
            (number_params*0.9, number_params*0.7).
        **kwargs : dict
            Keyword arguments passed to `~gammapy.visualization.plot_heatmap`.

        Returns
        -------
        ax : `~matplotlib.axes.Axes`, optional
            Matplotlib axes.

        """
        from gammapy.visualization import annotate_heatmap, plot_heatmap

        npars = len(self.parameters)
        figsize = (npars * 0.9, npars * 0.7) if figsize is None else figsize

        plt.figure(figsize=figsize)
        ax = plt.gca()
        kwargs.setdefault("cmap", "coolwarm")

        names = self.parameters.names
        im, cbar = plot_heatmap(
            data=self.correlation,
            col_labels=names,
            row_labels=names,
            ax=ax,
            vmin=-1,
            vmax=1,
            cbarlabel="Correlation",
            **kwargs,
        )
        annotate_heatmap(im=im)
        return ax

    @property
    def correlation(self):
        r"""Correlation matrix as a `numpy.ndarray`.

        Correlation :math:`C` is related to covariance :math:`\Sigma` via:

        .. math::
            C_{ij} = \frac{ \Sigma_{ij} }{ \sqrt{\Sigma_{ii} \Sigma_{jj}} }
        """
        err = np.sqrt(np.diag(self.data))

        with np.errstate(invalid="ignore", divide="ignore"):
            correlation = self.data / np.outer(err, err)

        return np.nan_to_num(correlation)

    @property
    def scipy_mvn(self):
        return scipy.stats.multivariate_normal(
            self.parameters.value, self.data, allow_singular=True
        )

    def __str__(self):
        return str(self.data)

    def __array__(self):
        return self.data


class CovarianceMixin:
    """Mixin class for covariance property on multi-components models"""

    def _check_covariance(self):
        if not self.parameters == self._covariance.parameters:
            self._covariance = Covariance.from_stack(
                [model.covariance for model in self._models]
            )

    @property
    def covariance(self):
        """Covariance as a `~gammapy.modeling.Covariance` object."""
        self._check_covariance()

        for model in self._models:
            self._covariance.set_subcovariance(model.covariance)

        return self._covariance

    @covariance.setter
    def covariance(self, covariance):
        self._check_covariance()
        self._covariance.data = covariance

        for model in self._models:
            subcovar = self._covariance.get_subcovariance(model.covariance.parameters)
            model.covariance = subcovar
