# Licensed under a 3-clause BSD style license - see LICENSE.rst

from typing import List, Optional
import numpy as np
from astropy.coordinates import SkyCoord
from pydantic import ValidationError, validator
from gammapy.data import GTI
from gammapy.utils.metadata import CreatorMetaData, MetaData

__all__ = ["FluxMetaData"]

SEDTYPE = ["dnde", "flux", "eflux", "e2dnde", "likelihood"]
FPFORMAT = ["gadf-sed", "lightcurve"]


class FluxMetaData(MetaData):
    """Metadata containing information about the FluxPoints and FluxMaps.

    Parameters
    ----------
    sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type.
    ul_conf : float, optional
        Confidence level used for the upper limit computation.
    n_sigma : float, optional
        Significance threshold above which upper limits should be used.
    n_sigma_ul : float, optional
        Sigma number used to compute the upper limits.
    sqrt_ts_threshold_ul : float, optional
        Threshold on the square root of the likelihood value above which upper limits should be used.
    gtis : list of `~astropy.table.Table`, optional
        List of used Good Time Intervals (GTIs).
    target_name : str, optional
        Name of the target.
    target_position : `~astropy.coordinates.SkyCoord`, optional
        Coordinates of the target.
    obs_ids : list of int, optional
        ID list of the used observations.
    dataset_names : list of str, optional
        Name list of the used datasets.
    instrument : str, optional
        Name of the instrument.
    creation : `~gammapy.utils.CreatorMetaData`, optional
        The creation metadata.
    optional : dict, optional
        additional optional metadata.
    """

    sed_type: Optional[str]  # Are these 5 fields really optional?
    ul_conf: Optional[float]
    n_sigma: Optional[float]
    n_sigma_ul: Optional[float]
    sqrt_ts_threshold_ul: Optional[float]
    gtis: Optional[List[GTI]]
    target_name: Optional[str]  # Are these 2 fields really optional?
    target_position: Optional[SkyCoord]
    obs_ids: Optional[List[int]]
    dataset_names: Optional[List[str]]  # Are these 2 fields really optional?
    instrument: Optional[str]
    creation: Optional[CreatorMetaData]  # Is this field really optional?
    optional: Optional[dict]

    @validator("target_position")
    def validate_position(cls, v):
        if v is None:
            return SkyCoord(np.nan, np.nan, unit="deg", frame="icrs")
        elif isinstance(v, SkyCoord):
            return v
        else:
            raise ValidationError(
                f"Incorrect position. Expect SkyCoord got {type(v)} instead."
            )

    @validator("sed_type")
    def validate_sed(cls, v):
        if v is None:
            # raise ValidationError(f"[sed_type] should be precised. Expect {SEDTYPE}")
            return None
        elif isinstance(v, str):
            if str not in SEDTYPE:
                raise ValidationError(f"Incorrect [sed_type]. Expect {SEDTYPE}")
            else:
                return v

    @validator("gtis")
    def validate_gtis(cls, v):
        if v is None:
            return [None]
        elif v is [None]:
            return v
        elif isinstance(v, List[GTI]):
            return v
        else:
            raise ValidationError(
                f"Incorrect [gtis]. Expect a list of GTIs, got {type(v)} instead."
            )

    @validator("creation")
    def validate_creation(cls, v):
        if v is None:
            raise ValidationError(
                f"[creation] should be precised. Expect {type(CreatorMetaData)}."
            )
        elif isinstance(v, CreatorMetaData):
            return v
        else:
            raise ValidationError(
                f"Incorrect [creation]. Expect 'CreatorMetaData' got {type(v)} instead."
            )

    @classmethod
    def from_default(cls):
        return cls(
            creation=CreatorMetaData.from_default(), target_position=None, gtis=None
        )

    @classmethod
    def to_header(self, format=None):
        """Store the FluxPoints metadata into a fits header.

        Parameters
        ----------
        format : {"gadf-sed", "lightcurve"}
            The header data format.

        Returns
        -------
        header : dict
            The header dictionary.
        """

        if format is None or format not in FPFORMAT:
            raise ValueError(
                f"Metadata creation with format {format} is not supported. Use {FPFORMAT}"
            )

        hdr_dict = self.creation.to_header()
        hdr_dict["SED_TYPE"] = self.sed_type
        # hdr_dict["UL_CONF"] =
        # ul_conf: Optional[float]  # Are these 4 fields really optional?
        # n_sigma: Optional[float]
        # n_sigma_ul: Optional[float]
        # sqrt_ts_threshold_ul: Optional[float]
        # gtis: Optional[List[GTI]]
        # target_name: Optional[str]
        # target_position: Optional[SkyCoord]
        # obs_ids: Optional[List[int]]
        # dataset_names: List[str]
        # instrument: Optional[str]

        return hdr_dict

    # @classmethod
    # def from_header(cls, hdr, format="gadf"):
    #     """Builds creator metadata from fits header.
    #     Parameters
    #     ----------
    #     hdr : dict
    #         the header dictionary
    #     format : str
    #         header format. Default is 'gadf'.
    #     """
    #     if format != "gadf":
    #         raise ValueError(f"Creator metadata: format {format} is not supported.")
    #
    #     from_header(hdr, format)

    @classmethod
    def from_dict(cls, data):
        """Extract metadata from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing data.
        """

        sed_type = data["SED_TYPE"] if "SED_TYPE" in data else None
        creation = CreatorMetaData.from_dict(data)

        ul_conf = data["ul_conf"] if "ul_conf" in data else None
        n_sigma = data["n_sigma"] if "n_sigma" in data else None
        n_sigma_ul = data["n_sigma_ul"] if "n_sigma_ul" in data else None
        sqrt_ts_threshold_ul = (
            data["sqrt_ts_threshold_ul"] if "sqrt_ts_threshold_ul" in data else None
        )
        target_name = data["target_name"] if "target_name" in data else None
        obs_ids = data["obs_ids"] if "obs_ids" in data else [None]
        dataset_names = data["dataset_names"] if "dataset_names" in data else [None]
        instrument = data["instrument"] if "instrument" in data else None
        target_position = data["target_position"] if "target_position" in data else None
        gtis = data["gtis"] if "gtis" in data else [None]

        print(gtis)
        meta = cls(
            sed_type=sed_type,
            creation=creation,
            ul_conf=ul_conf,
            n_sigma=n_sigma,
            n_sigma_ul=n_sigma_ul,
            sqrt_ts_threshold_ul=sqrt_ts_threshold_ul,
            target_name=target_name,
            target_position=target_position,
            dataset_names=dataset_names,
            instrument=instrument,
            obs_ids=obs_ids,
            gtis=gtis,
        )
        if "optional" in data:
            meta.optional = data["optional"]

        return meta
