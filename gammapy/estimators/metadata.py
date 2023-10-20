# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
from typing import List, Optional
import numpy as np
from scipy import stats
from astropy.coordinates import SkyCoord
from pydantic import ValidationError, validator
from gammapy.data import GTI
from gammapy.utils.metadata import CreatorMetaData, MetaData

__all__ = ["FluxMetaData"]

SEDTYPE = ["dnde", "flux", "eflux", "e2dnde", "likelihood"]
FPFORMAT = ["gadf-sed", "lightcurve"]

log = logging.getLogger(__name__)


class FluxMetaData(MetaData):
    """Metadata containing information about the FluxPoints and FluxMaps.

    Parameters
    ----------
    sed_type : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type.
    sed_type_init : {"dnde", "flux", "eflux", "e2dnde", "likelihood"}, optional
        SED type of the initial data.
    ul_conf : float, optional
        Confidence level used for the upper limit computation.
    n_sigma : float, optional
        Significance threshold above which upper limits should be used.
    n_sigma_ul : float, optional
        Sigma number used to compute the upper limits.
    sqrt_ts_threshold_ul : float, optional
        Threshold on the square root of the likelihood value above which upper limits should be used.
    gti : `~gammapy.data.gti`, optional
        used Good Time Intervals.
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

    sed_type: Optional[str]  # Are these 6 fields really optional?
    sed_type_init: Optional[str]
    ul_conf: Optional[float]
    n_sigma: Optional[float]
    n_sigma_ul: Optional[float]
    sqrt_ts_threshold_ul: Optional[float]
    gti: Optional[GTI]
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
    def validate_sed_type(cls, v):
        # if v is None:
        #     # raise ValidationError(f"[sed_type] should be precised. Expect {SEDTYPE}")
        #     return None
        # elif isinstance(v, str):
        #     if str not in SEDTYPE:
        #         raise ValidationError(f"Incorrect [sed_type]. Expect {SEDTYPE}")
        #     else:
        #         return v
        if isinstance(v, str):
            if str not in SEDTYPE:
                raise ValidationError(f"Incorrect [sed_type]. Expect {SEDTYPE}")
        return v

    @validator("sed_type_init")
    def validate_sed_type_init(cls, v):
        # if v is None:
        #     # raise ValidationError(f"[sed_type_init] should be precised. Expect {SEDTYPE}")
        #     return None
        # elif isinstance(v, str):
        #     if str not in SEDTYPE:
        #         raise ValidationError(f"Incorrect [sed_type_init]. Expect {SEDTYPE}")
        #     else:
        #         return v
        if isinstance(v, str):
            if str not in SEDTYPE:
                raise ValidationError(f"Incorrect [sed_type_init]. Expect {SEDTYPE}")
        return v

    @validator("gti")
    def validate_gti(cls, v):
        if v is None:
            return None
        elif isinstance(v, GTI):
            return v
        else:
            raise ValidationError(
                f"Incorrect [gti]. Expect a GTI, got {type(v)} instead."
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
            creation=CreatorMetaData.from_default(), target_position=None, gti=None
        )

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
        hdr_dict["SED_TYPE_INIT"] = self.sed_type_init
        hdr_dict["UL_CONF"] = self.ul_conf
        hdr_dict["N_SIGMA"] = self.n_sigma
        hdr_dict["N_SIGMA_UL"] = self.n_sigma_ul
        hdr_dict["SQRT_TS_THRESHOLD_UL"] = self.sqrt_ts_threshold_ul
        # hdr_dict["GTI"] = self.gti #They should be written in a HDU, in in the header
        hdr_dict["TARGET_NAME"] = self.target_name
        hdr_dict["TARGET_POSITION"] = self.target_position
        hdr_dict["OBS_IDS"] = self.obs_ids
        hdr_dict["DATASET_NAMES"] = self.dataset_names
        hdr_dict["INSTRUMENT"] = self.instrument
        # Do not forget that we have optional metadata

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
        sed_type_init = data["SED_TYPE_INIT"] if "SED_TYPE_INIT" in data else None
        creation = CreatorMetaData.from_dict(data)

        ul_conf = data["UL_CONF"] if "UL_CONF" in data else None
        n_sigma = data["N_SIGMA"] if "N_SIGMA" in data else None

        n_sigma_ul = data["N_SIGMA_UL"] if "N_SIGMA_UL" in data else None
        if ul_conf:
            if n_sigma_ul:
                if n_sigma_ul != np.round(stats.norm.isf(0.5 * (1 - ul_conf)), 1):
                    log.warn(
                        f"Inconsistency between n_sigma_ul={n_sigma_ul} and ul_conf={ul_conf}"
                    )
            else:
                n_sigma_ul = np.round(stats.norm.isf(0.5 * (1 - ul_conf)), 1)

        sqrt_ts_threshold_ul = (
            data["SQRT_TS_THRESHOLD_UL"] if "SQRT_TS_THRESHOLD_UL" in data else None
        )
        target_name = data["TARGET_NAME"] if "TARGET_NAME" in data else None
        obs_ids = data["OBS_IDS"] if "OBS_IDS" in data else [None]
        dataset_names = data["DATASET_NAMES"] if "DATASET_NAMES" in data else [None]
        instrument = data["INSTRUMENT"] if "INSTRUMENT" in data else None
        target_position = data["TARGET_POSITION"] if "TARGET_POSITION" in data else None
        # gti = data["gti"] if "gti" in data else None #There are stored in a dedicated HDU

        meta = cls(
            sed_type=sed_type,
            sed_type_init=sed_type_init,
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
            # gti=gti,
        )
        if "optional" in data:
            meta.optional = data["optional"]

        return meta

    def to_table(self, table):
        """Write the metadata into a data table.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Flux table.
        """

        table.meta["SED_TYPE"] = self.sed_type
        table.meta["SED_TYPE_INIT"] = self.sed_type_init
        table.meta["UL_CONF"] = self.ul_conf
        table.meta["N_SIGMA"] = self.n_sigma
        table.meta["N_SIGMA_UL"] = self.n_sigma_ul
        table.meta["SQRT_TS_THRESHOLD_UL"] = self.sqrt_ts_threshold_ul
        table.meta["TARGET_NAME"] = self.target_name
        table.meta["OBS_IDS"] = self.obs_ids
        table.meta["DATASET_NAMES"] = self.dataset_names
        table.meta["INSTRUMENT"] = self.instrument
        table.meta["TARGET_POSITION"] = self.target_position
        # table.meta["GTI"] = self.gti #There are stored in a dedicated HDU

        self.creation.to_table(table)

        # Ignore the optional metadata for the moment
