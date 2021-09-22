# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.datasets import Datasets
from gammapy.modeling import Fit
from .core import Estimator

log = logging.getLogger(__name__)

AVAILABLE_METHODS = [
    "ts",
    "err",
    "ul",
    "scan"
]


class ParameterEstimator(Estimator):
    """Model parameter estimator.

    Estimates a model parameter for a group of datasets. Compute best fit value,
    symmetric and delta TS for a given null value. Additionally asymmetric errors
    as well as parameter upper limit and fit statistic profile can be estimated.

    Parameters
    ----------
    n_sigma : int
        Sigma to use for asymmetric error computation. Default is 1.
    n_sigma_ul : int
        Sigma to use for upper limit computation. Default is 2.
    null_value : float
        Which null value to use for the parameter
    selection_optional : list of str
        Which additional quantities to estimate. Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors on parameter best fit value.
            * "ul": estimate upper limits.
            * "scan": estimate fit statistic profiles.

        Default is None so the optional steps are not executed.
    fit : `Fit`
        Fit instance specifying the backend and fit options.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    full_output: bool
        Print the messages of all fitters. Default is False
    """

    tag = "ParameterEstimator"
    _available_selection_optional = ["errn-errp", "ul", "scan"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        null_value=1e-150,
        selection_optional=None,
        fit=None,
        reoptimize=True,
        full_output=False, #BKH
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.null_value = null_value
        self.selection_optional = selection_optional
        self.full_output = full_output

        if fit is None:
            fit = Fit()

        self.fit = fit
        self.reoptimize = reoptimize

    def estimate_best_fit(self, datasets, parameter):
        """Estimate parameter asymmetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values. Entries are:

                * parameter.name: best fit parameter value
                * "stat": best fit total stat.
                * parameter.name_err: covariance-based error estimate on parameter value
                * "success": boolean flag for fit success
                * "message": message of the fitter
                * "ncalls": Number of minimizer calls
        """
        value, total_stat, success, error = np.nan, 0.0, False, np.nan
        # value, total_stat, success, message, error = np.nan, 0, False, "", np.nan
        # value, total_stat, success, message, error, ncalls = np.nan, 0, False, "", np.nan, 0

        if np.any(datasets.contributes_to_stat):
            result = self.fit.run(datasets=datasets)
            value, error = parameter.value, parameter.error
            total_stat = result.optimize_result.total_stat
            success = result.success
            # # message = result["optimize_result"].message BKH
            # total_stat = result["optimize_result"].total_stat
            # success = result["optimize_result"].success
            # message = result["optimize_result"].message
            # ncalls = result["optimize_result"].ncalls

        return {
            f"{parameter.name}": value,
            "stat": total_stat,
            f"{parameter.name}_err": error * self.n_sigma,
            "success": success,
            "message": message,
            "ncalls": ncalls,
        }

    def estimate_ts(self, datasets, parameter):
        """Estimate parameter ts

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the TS of the best fit value compared to the null hypothesis. Entries are:

                * "ts" : fit statistic difference with null hypothesis
                * "npred" : predicted number of counts per dataset
                # * "npred_null" : predicted number of counts per dataset in the null hypothesis BKH
                # * "success_ts": boolean flag for fit success
                # * "message_ts": message of the fitter
        """
        npred = self.estimate_npred(datasets=datasets)
        success, message = True, ""

        if not np.any(datasets.contributes_to_stat):
            stat = np.nan
            npred["npred"][...] = np.nan
            success = False
            message = "Dataset not contributing"
        else:
            stat = datasets.stat_sum()
            message = "Succeed to evaluate the TS"

        with datasets.parameters.restore_status():
            # compute ts value
            parameter.value = self.null_value

            if self.reoptimize:
                parameter.frozen = True
                res = self.fit.optimize(datasets=datasets)
                message = res.message
                success &= res.success

            ts = datasets.stat_sum() - stat

        return {
            "ts": ts,
            "npred": npred["npred"],
            "npred_null": npred_null["npred"],
            "success_ts": success,
            "message_ts": message,
        }

    def estimate_errn_errp(self, datasets, parameter):
        """Estimate parameter asymmetric errors

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            Datasets
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the parameter asymmetric errors. Entries are:

                * {parameter.name}_errp : positive error on parameter value
                * {parameter.name}_errn : negative error on parameter value
                # * "success_err": boolean flag for error success BKH
                # * "message_err": message of the fitter
        """
        if not np.any(datasets.contributes_to_stat):
            return {
                f"{parameter.name}_errp": np.nan,
                f"{parameter.name}_errn": np.nan,
                "success_err": False,
                "message_err": "Dataset not contributing",
            }

        self.fit.optimize(datasets=datasets)

        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma,
            reoptimize=self.reoptimize,
        )

        message = ""
        for k, msg in res.items():
            if k.startswith('message'):
                message += msg

        return {
            f"{parameter.name}_errp": res["errp"],
            f"{parameter.name}_errn": res["errn"],
            "success_err": res["fit_results"],
            "message_err": message,
        }

    def estimate_scan(self, datasets, parameter):
        """Estimate parameter stat scan.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the parameter fit scan values. Entries are:

                * parameter.name_scan : parameter values scan
                * "stat_scan" : fit statistic values scan
                * "success_scan": boolean flag for scan success
                * "message_scan": message of the fitter
        """
        scan_values = parameter.scan_values

        if not np.any(datasets.contributes_to_stat):
            return {
                f"{parameter.name}_scan": scan_values,
                "stat_scan": scan_values * np.nan,
                "success_scan": False, #BKH
                "message_scan": "Dataset not contributing",
            }

        self.fit.optimize(datasets=datasets)

        profile = self.fit.stat_profile(
            datasets=datasets, parameter=parameter, reoptimize=self.reoptimize
        )
        success, message = True, ""
        for res, mes in zip(profile["fit_results"], profile["message_prof"]):
            success &= res
            message += mes

        return {
            f"{parameter.name}_scan": scan_values,
            "stat_scan": profile["stat_scan"],
            "success_scan": success,
            "message_scan": message,
        }

    def estimate_ul(self, datasets, parameter):
        """Estimate parameter ul.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `Parameter`
            For which parameter to get the value

        Returns
        -------
        result : dict
            Dict with the parameter ULs. Entries are:

                * parameter.name_ul : upper limit on parameter value
                * "success_ul": boolean flag for upper limit success
                * "message_ul": message of the fitter
        """
        if not np.any(datasets.contributes_to_stat):
            return {
                f"{parameter.name}_ul": np.nan,
                "success_ul": False,
                "message_ul": "Dataset not contributing",
            }

        self.fit.optimize(datasets=datasets)

        res = self.fit.confidence(
            datasets=datasets,
            parameter=parameter,
            sigma=self.n_sigma_ul,
            reoptimize=self.reoptimize,
        )

        message = ""
        for k, msg in res.items():
            if k.startswith('message'):
                message += msg

        return {
            f"{parameter.name}_ul": res["errp"] + parameter.value,
            "success_ul": res["fit_results"],
            "message_ul": message,
        }

    @staticmethod
    def estimate_counts(datasets):
        """Estimate counts for the flux point.

        Parameters
        ----------
        datasets : Datasets
            Datasets

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with the sum of the
            masked counts.
        """
        counts = []

        for dataset in datasets:
            mask = dataset.mask
            counts.append(dataset.counts.data[mask].sum())

        return {
            "counts": np.array(counts, dtype=int),
            "datasets": datasets.names,
        }

    @staticmethod
    def estimate_npred(datasets):
        """Estimate npred for the flux point.

        Parameters
        ----------
        datasets : Datasets
            Datasets

        Returns
        -------
        result : dict
            Dict with an array with one entry per dataset with the sum of the
            masked npred.
        """
        npred = []

        for dataset in datasets:
            mask = dataset.mask
            npred.append(dataset.npred().data[mask].sum())

        return {
            "npred": np.array(npred),
            "datasets": datasets.names,
        }

    def run(self, datasets, parameter):
        """Run the parameter estimator.

        Parameters
        ----------
        datasets : `~gammapy.datasets.Datasets`
            The datasets used to estimate the model parameter
        parameter : `str` or `Parameter`
            For which parameter to run the estimator

        Returns
        -------
        result : dict
            Dict with the various parameter estimation values.
        """
        datasets = Datasets(datasets)
        parameter = datasets.parameters[parameter]
        result = {}

        with datasets.parameters.restore_status():

            if not self.reoptimize:
                datasets.parameters.freeze_all()
                parameter.frozen = False

            result.update(self.estimate_best_fit(datasets, parameter))
            result["fit_status"] = int(result["success"])

            result.update(self.estimate_ts(datasets, parameter))

            # If the fit has not converged, should we compute the other quantities?
            other_success = True
            if "errn-errp" in self.selection_optional:
                result.update(self.estimate_errn_errp(datasets, parameter))
                other_success &= result["success_err"]

            if "ul" in self.selection_optional:
                result.update(self.estimate_ul(datasets, parameter))
                other_success &= result["success_ul"]

            if "scan" in self.selection_optional:
                result.update(self.estimate_scan(datasets, parameter))
                other_success &= result["success_scan"]

        if result["fit_status"] == 1 and other_success:
            result["fit_status"] = 2

        result.update(self.estimate_counts(datasets))

        if self.full_output:
            log.info(result["message"])
        for method in AVAILABLE_METHODS:
            if f"message_{method}" not in result:
                continue
            if not self.full_output:
                del result[f"success_{method}"], result[f"message_{method}"]
            else:
                log.info(result[f"message_{method}"])

        return result
