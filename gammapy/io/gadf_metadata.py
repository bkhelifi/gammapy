# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import Literal, Optional, Union

from astropy.io import fits
from astropy.table import Table
from astropy.units import UnitTypeError
from pydantic import BaseModel, model_validator, ConfigDict, Field, ValidationInfo
from typing import ClassVar

# from gammapy.datasets.io import OGIPDatasetReader, OGIPDatasetWriter
from gammapy.data.io import EventListReader
from gammapy.data import GTI
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.irf import (
    EffectiveAreaTable2D,
    EnergyDispersion2D,
    BackgroundIRF,
    Background2D,
    Background3D,
    ParametricPSF,
    PSF3D,
    EnergyDependentMultiGaussPSF,
    PSFKing,
    RadMax2D,
)
from gammapy.irf.io import _get_hdu_type_and_class
from gammapy.utils.scripts import make_path
from gammapy.utils.table_validator import TableValidator

import logging

log = logging.getLogger(__name__)

# Column-definition schemes for the HDUs

GADF_EVENT_TABLE_DEFINITION = """
EVENT_ID:     { dtype: int,   required: true, unit: null }
TIME:         { dtype: float, required: true, unit: s }
RA:           { dtype: float, required: true, unit: deg }
DEC:          { dtype: float, required: true, unit: deg }
ENERGY:       { dtype: float, required: true, unit: TeV }
EVENT_TYPE:   { dtype: int8 }
MULTIP:       { dtype: int }
GLON:         { dtype: float, unit: deg }
GLAT:         { dtype: float, unit: deg }
ALT:          { dtype: float, unit: deg }
AZ:           { dtype: float, unit: deg }
DETX:         { dtype: float, unit: deg }
DETY:         { dtype: float, unit: deg }
THETA:        { dtype: float, unit: deg }
PHI:          { dtype: float, unit: deg }
GAMMANESS:    { dtype: float }
DIR_ERR:      { dtype: float, unit: deg }
ENERGY_ERR:   { dtype: float, unit: TeV }
COREX:        { dtype: float, unit: m }
COREY:        { dtype: float, unit: m }
CORE_ERR:     { dtype: float, unit: m }
XMAX:         { dtype: float, unit: m }
XMAX_ERR:     { dtype: float, unit: m }
HIL_MSW:      { dtype: float, unit: '' }
HIL_MSW_ERR:  { dtype: float, unit: '' }
HIL_MSL:      { dtype: float, unit: '' }
HIL_MSL_ERR:  { dtype: float, unit: '' }
"""

GADF_GTI_TABLE_DEFINITION = """
START: { dtype: float, required: true, unit: s }
STOP:  { dtype: float, required: true, unit: s }
"""

GADF_POINTING_TABLE_DEFINITION = """
TIME:         { dtype: float, required: true, unit: s }
RA_PNT:       { dtype: float, required: true, unit: deg }
DEC_PNT:      { dtype: float, required: true, unit: deg }
ALT_PNT:      { dtype: float, unit: deg }
AZ_PNT:       { dtype: float, unit: deg }
"""

GADF_AEFF_2D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
EFFAREA:        { dtype: float, required: true, unit: m2, ndim: 2 }
"""

GADF_EDISP_2D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
MIGRA_LO:       { dtype: float, required: true, unit: '', ndim: 1 }
MIGRA_HI:       { dtype: float, required: true, unit: '', ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
MATRIX:         { dtype: float, required: true, unit: '', ndim: 3 }
"""

GADF_PSF_2D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
RAD_LO:         { dtype: float, required: true, unit: deg, ndim: 1 }
RAD_HI:         { dtype: float, required: true, unit: deg, ndim: 1 }
RPSF:           { dtype: float, required: true, unit: sr-1, ndim: 3 }
"""

GADF_PSF_3D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
SCALE:          { dtype: float, required: true, unit: sr-1, ndim: 1}
SIGMA_1:        { dtype: float, required: true, unit: deg, ndim: 2 }
SIGMA_2:        { dtype: float, required: true, unit: deg, ndim: 2}
SIGMA_3:        { dtype: float, required: true, unit: deg, ndim: 2 }
AMPL_2:         { dtype: float, required: true, unit: '', ndim: 2 }
AMPL_3:         { dtype: float, required: true, unit: '', ndim: 2 }
"""

GADF_PSF_KING_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
GAMMA:          { dtype: float, required: true, unit: '', ndim: 2 }
SIGMA:          { dtype: float, required: true, unit: '', ndim: 2 }
"""

GADF_BKG_2D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
BKG:            { dtype: float, required: true, unit: s-1 MeV-1 sr-1, ndim: 2 }
"""

GADF_BKG_3D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
DETX_LO:        { dtype: float, required: true, unit: deg, ndim: 1 }
DETX_HI:        { dtype: float, required: true, unit: deg, ndim: 1 }
DETY_LO:        { dtype: float, required: true, unit: deg, ndim: 1 }
DETY_HI:        { dtype: float, required: true, unit: deg, ndim: 1 }
BKG:            { dtype: float, required: true, unit: s-1 MeV-1 sr-1, ndim: 3 }
"""

GADF_RAD_MAX_2D_TABLE_DEFINITION = """
ENERG_LO:       { dtype: float, required: true, unit: TeV, ndim: 1 }
ENERG_HI:       { dtype: float, required: true, unit: TeV, ndim: 1 }
THETA_LO:       { dtype: float, required: true, unit: deg, ndim: 1 }
THETA_HI:       { dtype: float, required: true, unit: deg, ndim: 1 }
RAD_MAX:        { dtype: float, required: true, unit: deg, ndim: 2 }
"""

# The key is the class tag.
# TODO: extend the info here with the minimal header info
GADF_IRF_DL3_HDU_SPECIFICATION = {
    "bkg_3d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "BKG_3D",
            "FOVALIGN": "RADEC",
        },
    },
    "bkg_2d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "BKG_2D",
        },
    },
    "edisp_2d": {
        "extname": "ENERGY DISPERSION",
        "column_name": "MATRIX",
        "mandatory_keywords": {
            "HDUCLAS2": "EDISP",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "EDISP_2D",
        },
    },
    "psf_table": {
        "extname": "PSF_2D_TABLE",
        "column_name": "RPSF",
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_TABLE",
        },
    },
    "psf_3gauss": {
        "extname": "PSF_2D_GAUSS",
        "column_name": {
            "sigma_1": "SIGMA_1",
            "sigma_2": "SIGMA_2",
            "sigma_3": "SIGMA_3",
            "scale": "SCALE",
            "ampl_2": "AMPL_2",
            "ampl_3": "AMPL_3",
        },
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_3GAUSS",
        },
    },
    "psf_king": {
        "extname": "PSF_2D_KING",
        "column_name": {
            "sigma": "SIGMA",
            "gamma": "GAMMA",
        },
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "PSF_KING",
        },
    },
    "aeff_2d": {
        "extname": "EFFECTIVE AREA",
        "column_name": "EFFAREA",
        "mandatory_keywords": {
            "HDUCLAS2": "EFF_AREA",
            "HDUCLAS3": "FULL-ENCLOSURE",  # added here to have HDUCLASN in order
            "HDUCLAS4": "AEFF_2D",
        },
    },
    "rad_max_2d": {
        "extname": "RAD_MAX",
        "column_name": "RAD_MAX",
        "mandatory_keywords": {
            "HDUCLAS2": "RAD_MAX",
            "HDUCLAS3": "POINT-LIKE",
            "HDUCLAS4": "RAD_MAX_2D",
        },
    },
}


GADF_IRF_MAP_HDU_SPECIFICATION = {
    "edisp_kernel_map": "edisp",
    "edisp_map": "edisp",
    "psf_map": "psf",
    "psf_map_reco": "psf",
}

PRODUCT_MODELS = {
    "EVENTS": EventListReader,
    "GTI": GTI,
    "POINTING": None,
    "AEFF_2D": EffectiveAreaTable2D,
    "EDISP_2D": EnergyDispersion2D,
    "PSF_TABLE": PSF3D,
    "PSF_3D": PSF3D,
    "PSF_3GAUSS": EnergyDependentMultiGaussPSF,
    "PSF_KING": PSFKing,
    "BKG_2D": Background2D,
    "BKG_3D": Background3D,
    "RAD_MAX_2D": RadMax2D,
    "MapDataset": MapDataset,
    "SpectrumDataset": SpectrumDataset,
}

# Nested registry: version -> HDU -> column-definition YAML.
# Need to check the differences between versions

GADF_PRODUCTS_TABLE_DEFINITION = {
    "0.2": {
        "EVENTS": GADF_EVENT_TABLE_DEFINITION,
        "GTI": GADF_GTI_TABLE_DEFINITION,
        "POINTING": GADF_POINTING_TABLE_DEFINITION,
        "AEFF_2D": GADF_AEFF_2D_TABLE_DEFINITION,
        "EDISP_2D": GADF_EDISP_2D_TABLE_DEFINITION,
        "PSF_TABLE": GADF_PSF_2D_TABLE_DEFINITION,
        "PSF_3D": GADF_PSF_3D_TABLE_DEFINITION,
        "PSF_3GAUSS": GADF_PSF_3D_TABLE_DEFINITION,
        "PSF_KING": GADF_PSF_KING_TABLE_DEFINITION,
        "BKG_2D": GADF_BKG_2D_TABLE_DEFINITION,
        "BKG_3D": GADF_BKG_3D_TABLE_DEFINITION,
        "RAD_MAX_2D": GADF_RAD_MAX_2D_TABLE_DEFINITION,
    },
    "0.3": {
        "EVENTS": GADF_EVENT_TABLE_DEFINITION,
        "GTI": GADF_GTI_TABLE_DEFINITION,
        "POINTING": GADF_POINTING_TABLE_DEFINITION,
        "AEFF_2D": GADF_AEFF_2D_TABLE_DEFINITION,
        "EDISP_2D": GADF_EDISP_2D_TABLE_DEFINITION,
        "PSF_TABLE": GADF_PSF_2D_TABLE_DEFINITION,
        "PSF_3GAUSS": GADF_PSF_3D_TABLE_DEFINITION,
        "PSF_3D": GADF_PSF_3D_TABLE_DEFINITION,
        "PSF_KING": GADF_PSF_KING_TABLE_DEFINITION,
        "BKG_2D": GADF_BKG_2D_TABLE_DEFINITION,
        "BKG_3D": GADF_BKG_3D_TABLE_DEFINITION,
        "RAD_MAX_2D": GADF_RAD_MAX_2D_TABLE_DEFINITION,
    },
}

GADF_HDUDOC = {
    "0.1": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
    "0.2": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
    "0.3": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
}

GADF_DEFAULT_VERSION = "0.3"


DEFAULT_STRICT_READ = False
DEFAULT_STRICT_WRITE = True


# --------------- FORMAT AGNOSTIC BASE FIELDS ---------------
# Also used in other formats (like OGIP), separated here in anticipation
# The fields are written as optional but they are required at validation


class HDUFields(BaseModel):
    HDUCLASS: Optional[str] = "GADF"
    HDUVERS: Optional[str] = GADF_DEFAULT_VERSION
    HDUDOC: Optional[str] = None
    HDUCLAS1: Optional[str] = None


class HDUResponseFields(BaseModel):
    HDUCLAS2: Optional[str] = None
    HDUCLAS3: Optional[str] = None
    HDUCLAS4: Optional[str] = None


class GeneralFields(BaseModel):
    TELESCOP: Optional[str] = None
    INSTRUME: Optional[str] = None
    ORIGIN: Optional[str] = None
    CREATOR: Optional[str] = None


# --------------- GADF MANDATORY BASE FIELDS ---------------
# The fields are written as optional but they are required at validation


class GADFTimeFields(BaseModel):
    MJDREFI: Optional[int] = None
    MJDREFF: Optional[float] = None
    TIMEUNIT: Optional[str] = None
    TIMESYS: Optional[str] = None
    TIMEREF: Optional[str] = None
    TSTART: Optional[float] = None
    TSTOP: Optional[float] = None


class GADFEarthLocationFields(BaseModel):
    GEOLON: Optional[float] = None
    GEOLAT: Optional[float] = None
    ALTITUDE: Optional[float] = None


class GADFPointingFields(BaseModel):
    RA_PNT: Optional[float] = None
    DEC_PNT: Optional[float] = None
    ALT_PNT: Optional[float] = None
    AZ_PNT: Optional[float] = None


class GADFObsFields(BaseModel):
    OBS_ID: Optional[Union[int, str]] = None
    OBS_MODE: Optional[str] = None
    TASSIGN: Optional[str] = None
    ONTIME: Optional[float] = None
    LIVETIME: Optional[float] = None
    DATE_OBS: Optional[str] = Field(None, alias="DATE-OBS")
    TIME_OBS: Optional[str] = Field(None, alias="TIME-OBS")
    DATE_END: Optional[str] = Field(None, alias="DATE-END")
    TIME_END: Optional[str] = Field(None, alias="TIME-END")


# --------------- GENERAL BASE HEADERS ---------------
def _field_names(*models):
    return frozenset(name for m in models for name in m.model_fields)


class HDUHeader(BaseModel):
    """Base HDU header model."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")
    HDUCLASS: Optional[str] = "CUSTOM"
    REQUIRED: ClassVar[frozenset] = frozenset()

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name)

    def to_header(self):
        return {
            k: v for k, v in self.model_dump(by_alias=True).items() if v is not None
        }

    @classmethod
    def from_header(cls, header, strict=False):
        kwargs = {
            k: header[cls.model_fields[k].alias or k]
            for k in cls.model_fields
            if (cls.model_fields[k].alias or k) in header
        }
        return cls.model_validate(kwargs, context={"strict": strict})

    def _check_required(self):
        """Return list of missing mandatory FITS keywords (empty if none)."""
        return [
            self.model_fields[name].alias or name
            for name in self.REQUIRED
            if getattr(self, name) is None
        ]

    @model_validator(mode="after")
    def _enforce(self, info: ValidationInfo):
        strict = (info.context or {}).get("strict", False)
        missing = self._check_required()
        if missing:
            msg = f"Missing mandatory keyword(s): {missing}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return self


class HDUProductHeader(HDUHeader, GeneralFields):
    HDUCLAS1: Literal["PRODUCT"] = "PRODUCT"
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, GeneralFields)


# --------------- GADF BASE HEADER ---------------
class GADFHDUHeader(HDUFields, HDUHeader):
    HDUCLASS: Literal["GADF"] = "GADF"

    @model_validator(mode="after")
    def _enforce_gadf(self, info: ValidationInfo):
        strict = (info.context or {}).get("strict", False)
        if self.HDUVERS is not None and self.HDUVERS not in GADF_HDUDOC:
            msg = f"Non-standard HDUVERS {self.HDUVERS!r}; known: {list(GADF_HDUDOC)}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return self


# --------------- GADF EVENTS HEADER ---------------
class GADFEventsHeader(
    GADFHDUHeader,
    GeneralFields,
    GADFObsFields,
    GADFTimeFields,
    GADFEarthLocationFields,
    GADFPointingFields,
):
    HDUCLAS1: Literal["EVENTS"] = "EVENTS"
    DEADC: Optional[float] = None
    EQUINOX: Optional[Union[float, str]] = None
    RADECSYS: Optional[str] = None

    REQUIRED: ClassVar[frozenset] = _field_names(
        HDUFields,
        GeneralFields,
        GADFObsFields,
        GADFTimeFields,
        GADFEarthLocationFields,
        GADFPointingFields,
    ) | {"DEADC", "EQUINOX", "RADECSYS"}

    # Optional
    OBJECT: Optional[str] = None
    RA_OBJ: Optional[float] = None
    DEC_OBJ: Optional[float] = None
    OBSERVER: Optional[str] = None
    EV_CLASS: Optional[Union[int, str]] = None
    TELAPSE: Optional[float] = None
    TELLIST: Optional[Union[str, list]] = None
    N_TELS: Optional[int] = None
    TASSIGN: Optional[str] = None
    DST_VER: Optional[Union[int, str]] = None
    ANA_VER: Optional[Union[int, str]] = None
    CAL_VER: Optional[Union[int, str]] = None
    CONV_DEP: Optional[float] = None
    CONV_RA: Optional[float] = None
    CONV_DEC: Optional[float] = None
    TRGRATE: Optional[float] = None
    ZTRGRATE: Optional[float] = None
    MUONEFF: Optional[float] = None
    BROKPIX: Optional[float] = None
    AIRTEMP: Optional[float] = None
    PRESSURE: Optional[float] = None
    RELHUM: Optional[float] = None
    NSBLEVEL: Optional[float] = None

    @model_validator(mode="after")
    def _enforce_events(self, info: ValidationInfo):
        if not (info.context or {}).get("strict"):
            return self
        if isinstance(self.TELLIST, list):
            raise ValueError(
                f"TELLIST must be a comma-separated string, got {self.TELLIST!r}"
            )
        return self


# --------------- GADF GTI HEADER ---------------
class GADFGTIHeader(GADFHDUHeader, GADFTimeFields):
    HDUCLAS1: Literal["GTI"] = "GTI"
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, GADFTimeFields)


# --------------- GADF POINTING HEADER ---------------
class GADFPointingHeader(GADFHDUHeader, GADFTimeFields):
    HDUCLAS1: Literal["POINTING"] = "POINTING"
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, GADFTimeFields)


# --------------- GADF IRF HEADER ---------------
GADF_IRF_TAG_REQUIRED_FIELDS = {
    "aeff_2d": ["LO_THRES", "HI_THRES"],
    "bkg_2d": ["FOVALIGN"],
    "bkg_3d": ["FOVALIGN"],
}


class GADFResponseHeader(GADFHDUHeader, HDUResponseFields):
    HDUCLAS1: Literal["RESPONSE"] = "RESPONSE"
    EXTNAME: Optional[str] = None
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, HDUResponseFields) | {
        "EXTNAME"
    }

    OBS_ID: Optional[int] = None
    LO_THRES: Optional[float] = None
    HI_THRES: Optional[float] = None
    FOVALIGN: Optional["str"] = None

    def validate_for_tag(self, tag):
        """Check the tag-conditional mandatory keywords are present."""
        missing = [
            key
            for key in GADF_IRF_TAG_REQUIRED_FIELDS.get(tag, [])
            if getattr(self, key) is None
        ]
        if missing:
            raise ValueError(f"IRF tag {tag!r} requires {missing} but they are unset.")
        return self

    @classmethod
    def from_tag(cls, tag, **overrides):
        """Build a header from an IRF tag (e.g. 'aeff_2d').

        Pulls EXTNAME and the mandatory HDUCLASN keywords from
        IRF_DL3_HDU_SPECIFICATION; `overrides` supply the remaining
        descriptive metadata (TELESCOP, etc.).
        """
        if tag not in GADF_IRF_DL3_HDU_SPECIFICATION:
            raise ValueError(
                f"Unknown IRF tag {tag!r}; known: {list(GADF_IRF_DL3_HDU_SPECIFICATION)}"
            )
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        kwargs = {
            "EXTNAME": spec["extname"],
            **spec["mandatory_keywords"],
            **overrides,
        }
        return cls.from_header(kwargs)

    def apply_tag(self, tag):
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        self.EXTNAME = spec["extname"]
        for key, value in spec["mandatory_keywords"].items():
            if key in type(self).model_fields:
                setattr(self, key, value)
        return self


GADF_HEADER_MODELS = {
    "EVENTS": GADFEventsHeader,
    "GTI": GADFGTIHeader,
    "POINTING": GADFPointingHeader,
    "RESPONSE": GADFResponseHeader,
}


HEADER_MODELS = {
    "GADF": GADF_HEADER_MODELS,
}


GADF_MODELS = {
    "TABLE": GADF_PRODUCTS_TABLE_DEFINITION,
    "HEADER": GADF_HEADER_MODELS,
}

# This leaves room to add OGIP or other models properly
# By default, any format other than GADF is considered custom for now
DATA_FORMATS_MODELS = {
    "GADF": GADF_MODELS,
}


# CUSTOM HDU HEADER


class CustomHDUHeader(HDUHeader, HDUResponseFields, GeneralFields):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_header(cls, header):
        return cls.model_construct(**dict(header))


# --------------- BASE READER/WRITER ---------------

DEFAULT_DATA_FORMAT = "GADF"
DEFAULT_DATA_FORMAT_VERSION = GADF_DEFAULT_VERSION


def _header_key(meta):
    return meta.get("HDUCLAS1")


def _hdu_class_key(meta):
    hdu_class = meta.get("HDUCLAS4") or meta.get("HDUCLAS1")
    if hdu_class not in PRODUCT_MODELS.keys():
        hdu_class = meta.get("HDUCLAS2", "unknown")
        if (hdu_class in ["EFF_AREA", "RPSF", "EDISP", "BKG"]) and meta.get("HDUCLAS4"):
            _, hdu_class = _get_hdu_type_and_class(meta)  # CTA-1DC workaround
    return hdu_class.upper()


def _check_data_format(meta, format, strict):
    meta_format = meta.get("HDUCLASS")
    if meta_format != format:
        if strict:
            raise ValueError(
                f"HDUReaderWriter objects expected a {format} HDU class, {meta_format} was provided."
            )
    return meta_format


class UnknownHDUClass(IOError):
    """Raised when a file contains an unknown HDUCLASS."""


class HDUReaderWriter:
    """IO class for reading and writing HDUs, dispatched by (format, version, HDU). Default data format is GADF v0.3"""

    DEFAULT_HDU = None

    def __init__(
        self,
        table,
        header,
        data=None,
        hdu=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
    ):
        hdu = hdu or self.DEFAULT_HDU
        if hdu is None:
            raise ValueError(
                "HDUReaderWriter objects require an `hdu` (no default for this class)."
            )
        self.table = table
        self.header = header
        self.data = data
        self.hdu = hdu
        self.format = format
        self.version = version

    @classmethod
    def table_validator(
        cls, hdu, format=DEFAULT_DATA_FORMAT, version=DEFAULT_DATA_FORMAT_VERSION
    ):
        if format in DATA_FORMATS_MODELS.keys():
            definition = DATA_FORMATS_MODELS[format]["TABLE"][version][hdu]
            return TableValidator.from_yaml(definition)
        else:
            log.warning(f"No table definition for {format}")

    @classmethod
    def format_validator(cls, header, table, format, version, strict):
        errors = []
        try:
            cls.table_validator(
                hdu=_hdu_class_key(table.meta), format=format, version=version
            ).run(table)
        except (KeyError, TypeError, UnitTypeError) as e:  # the types run() raises
            errors.append(f"table: {e}")

        if errors:
            msg = f"Metadata are not {format} compliant: " + ", ".join(errors)
            if strict:
                raise ValueError(msg)
            log.warning(
                "%s\n"
                + f"{format} compliance not enforced, please ensure FAIR principles.",
                msg,
            )
        return not errors

    @classmethod
    def read(
        cls,
        filename,
        hdu=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
        verbose=True,
    ):
        hdu = hdu or cls.DEFAULT_HDU
        if hdu is None:
            raise ValueError(
                "HDUReaderWriter objects require an `hdu` (no default for this class)."
            )
        filename = make_path(filename)
        with fits.open(filename, memmap=False) as hdulist:
            fits_hdu = hdulist[hdu]
            meta = dict(fits_hdu.header)
            meta_format = _check_data_format(meta, format, strict)

            if fits_hdu.is_image:
                header = CustomHDUHeader.from_header(meta)
                print(f"{hdu}: {meta_format}, data")
                return cls(
                    table=None,
                    header=header,
                    data=fits_hdu.data,
                    hdu=hdu,
                    format=meta_format,
                    version=None,
                )
            table = Table.read(fits_hdu)
            data = None
            meta = table.meta
            hdu_class = _hdu_class_key(meta)
            meta_format = _check_data_format(meta, format, strict)
            print(f"{hdu}: {meta_format}, table")

            if meta_format in DATA_FORMATS_MODELS.keys():
                try:
                    header = DATA_FORMATS_MODELS[meta_format]["HEADER"][
                        _header_key(meta)
                    ].from_header(meta, strict)
                    cls.format_validator(header, table, meta_format, version, strict)
                except (ValueError, KeyError, TypeError, UnitTypeError) as e:
                    if strict:
                        raise
                    if verbose:
                        log.warning(
                            "Header validation failed (%s), using CustomHDUHeader.", e
                        )

                    header = CustomHDUHeader.from_header(meta)
                    version = None
            else:
                header = CustomHDUHeader.from_header(meta)
                version = None

            return cls(
                table=table,
                header=header,
                data=data,
                hdu=hdu_class,
                format=meta_format,
                version=version,
            )

    def to_product(self):
        if self.hdu in PRODUCT_MODELS.keys():
            try:
                return PRODUCT_MODELS[self.hdu].from_table(self.table)
            except (ValueError, AttributeError) as e:
                log.warning(e)

    def to_table_hdu(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        """Export to a `~astropy.io.fits.BinTableHDU`."""
        self.format_validator(self.header, self.table, format, version, strict)
        table_hdu = fits.BinTableHDU(self.table, name=self.hdu)
        table_hdu.header.update(self.header.to_header())
        return table_hdu

    def to_hdulist(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        hdulist = [fits.PrimaryHDU(), self.to_table_hdu(format, version, strict)]
        return hdulist

    def write(
        self,
        filename,
        overwrite=False,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        hdulist = self.to_hdulist(format, version, strict)
        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )


# --------------- EVENTS READER/WRITER ---------------
class GADFEventsReaderWriter(HDUReaderWriter):
    """IO class specialised to the EVENTS HDU."""

    DEFAULT_HDU = "EVENTS"
    format = "GADF"

    @classmethod
    def from_eventlist(
        cls, eventlist, version=GADF_DEFAULT_VERSION, strict=DEFAULT_STRICT_READ
    ):
        data = None
        table = eventlist.table
        header = GADF_HEADER_MODELS[cls.DEFAULT_HDU].from_header(table.meta, strict)
        cls.format_validator(header, table, cls.format, version, strict)
        return cls(
            table=table, header=header, data=data, hdu=cls.DEFAULT_HDU, version=version
        )


# --------------- GTI READER/WRITER ---------------
class GADFGTIReaderWriter(HDUReaderWriter):
    """IO class specialised to the GTI HDU."""

    DEFAULT_HDU = "GTI"


# --------------- IRF READER/WRITER ---------------
def resolve_irf_tag(meta):
    """Identify the IRF tag from a RESPONSE HDU header via HDUCLAS4."""
    hduclas4 = meta.get("HDUCLAS4")
    if hduclas4 is None:
        raise ValueError("Missing HDUCLAS4: cannot identify the IRF type.")
    tag = hduclas4.lower()
    if tag not in GADF_IRF_DL3_HDU_SPECIFICATION:
        raise ValueError(
            f"Unknown IRF HDUCLAS4={hduclas4!r}; known: {list(GADF_IRF_DL3_HDU_SPECIFICATION)}"
        )
    return tag


class GADFResponseReaderWriter(HDUReaderWriter):
    """IO for GADF RESPONSE HDUs.

    Header/metadata is handled here. The binned data (axes + payload) is
    delegated to the gammapy IRF object's own from_table / to_table.
    """

    GADF_HEADER_MODEL = GADFResponseHeader

    @classmethod
    def from_irf(cls, irf, version=GADF_DEFAULT_VERSION, strict=DEFAULT_STRICT_READ):
        """Build from an in-memory gammapy IRF object."""
        table = irf.to_table()
        tag = irf.tag if isinstance(irf.tag, str) else irf.tag[0]
        if strict:
            cls.table_validator(tag.upper(), version).run(table)
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        meta = {**dict(getattr(irf, "meta", {}) or {}), **spec["mandatory_keywords"]}
        header = cls.GADF_HEADER_MODEL.from_header(meta, strict)
        return cls(
            table=table,
            header=header,
            data=None,
            hdu=spec["extname"],
            format="GADF",
            version=version,
        )


GADF_READER_WRITER_MODELS = {
    "EVENTS": GADFEventsReaderWriter,
    "GTI": GADFGTIReaderWriter,
    "RESPONSE": GADFResponseReaderWriter,
}


DATA_FORMATS_MODELS["GADF"]["READER_WRITER"] = GADF_READER_WRITER_MODELS


class HDUListReaderWriter:
    def __init__(self, hdu_dict, format_list):
        self.hdu_dict = hdu_dict
        self.format_list = format_list

    @staticmethod
    def _hdu_kind(hdu):
        if isinstance(hdu, fits.PrimaryHDU):
            return "primary"
        return "image" if hdu.is_image else "table"

    @classmethod
    def _read_hdu(cls, hdu, format, version, strict, verbose=True):
        kind = cls._hdu_kind(hdu)

        if kind == "primary":  # no HDUCLASS in primary
            header = HDUHeader.from_header(dict(hdu.header), strict)
            return HDUReaderWriter(
                data=None,
                table=None,
                header=header,
                hdu=hdu.name,
                format=format,
                version=version,
            )

        if kind == "image":  # no table validation
            meta, data, table = dict(hdu.header), hdu.data, None
            hkey, validate = "IMAGE", False
        else:  # table
            table = Table.read(hdu)
            meta, data = dict(table.meta), None
            hkey, validate = _header_key(meta), True

        hdu_class = _hdu_class_key(meta)
        meta_format = _check_data_format(meta, format, strict)
        models = DATA_FORMATS_MODELS.get(meta_format)

        if models is None:  # unknown format -> custom, no version
            header = CustomHDUHeader.from_header(meta)
            return HDUReaderWriter(
                data=data,
                table=table,
                header=header,
                hdu=hdu_class,
                format=meta_format,
                version=None,
            )

        header = models["HEADER"][hkey].from_header(meta, strict)
        ReaderWriter = models.get("READER_WRITER", {}).get(hkey, HDUReaderWriter)
        if validate:
            try:
                ReaderWriter.format_validator(
                    header, table, meta_format, version, strict
                )
            except (ValueError, KeyError, TypeError, UnitTypeError) as e:
                if strict:
                    raise
                if verbose:
                    log.warning(
                        "Header validation failed (%s); %s v%s not enforced.",
                        e,
                        meta_format,
                        version,
                    )
        return ReaderWriter(
            data=data,
            table=table,
            header=header,
            hdu=hdu_class,
            format=meta_format,
            version=version,
        )

    @classmethod
    def read(
        cls,
        filename,
        format=DEFAULT_DATA_FORMAT,
        version=GADF_DEFAULT_VERSION,
        strict=DEFAULT_STRICT_READ,
        verbose=True,
    ):
        filename = make_path(filename)
        if verbose:
            log.warning(
                "Reading %s \nExpected data format: %s, strict=%s",
                filename,
                format,
                strict,
            )

        hdu_dict, format_list = {}, []
        with fits.open(filename, memmap=False) as hdulist:
            for hdu in hdulist:
                rw = cls._read_hdu(hdu, format, version, strict)
                hdu_dict[hdu.name] = rw
                format_list.append(rw.format)
            if verbose:
                log.warning("Detected formats: %s", format_list)
        return cls(hdu_dict=hdu_dict, format_list=format_list)

    def to_product_dict(cls):
        product_dict = {}
        for hdu_name in list(cls.hdu_dict.keys())[1:]:
            if hdu_name != "PRIMARY":
                product_dict[cls.hdu_dict[hdu_name].hdu] = cls.hdu_dict[
                    hdu_name
                ].to_product()
        return product_dict

    def to_hdulist(
        cls,
        format=DEFAULT_DATA_FORMAT,
        version=GADF_DEFAULT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        hdulist = [cls.hdu_dict["PRIMARY"]]
        for hdu in list(cls.hdu_dict.keys())[1:]:
            hdulist.append(cls.hdu_dict[hdu].to_table_hdu(format, version, strict))
        return hdulist

    # Having a doubt here, should I be always declaring a new primary hdu like it is done elsewhere?
    # Because if I want to write a hdulist combining hdus from separate files, like one with the events and another one with the IRFs, I lose the original information although I am not creating or modifying anything


class ProductReaderWriter:
    """IO for multi-HDU gammapy products (DL4/DL5).

    Data reconstruction is delegated to the gammapy object's own
    from_hdulist / to_hdulist. This layer owns only the shared (primary)
    metadata: validation on read, GADF-Plus keywords on write.
    """

    PRODUCT = None  # subclass sets, e.g. "MapDataset"

    def __init__(
        self,
        product,
        header=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
    ):
        self.product = product  # the gammapy object (MapDataset, ...)
        self.header = header  # product-level metadata model (or CustomHDUHeader)
        self.format = format
        self.version = version

    @classmethod
    def read(
        cls,
        filename,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
    ):
        filename = make_path(filename)
        primary = fits.getheader(str(filename))  # primary-only, no data
        meta_format = _check_data_format(primary, format, strict)
        try:
            header = HDUProductHeader.from_header(dict(primary), strict)
        except (ValueError, KeyError, TypeError) as e:
            if strict:
                raise
            log.warning("Product header validation failed (%s).", e)
            # header = CustomHDUHeader.from_header(dict(primary))

        product = PRODUCT_MODELS[cls.PRODUCT].read(filename)
        return cls(product=product, header=header, format=meta_format, version=version)

    def write(
        self, filename, overwrite=False, checksum=False, strict=DEFAULT_STRICT_WRITE
    ):
        # delegate assembly to gammapy, then overlay validated shared metadata
        hdulist = self.product.to_hdulist()
        if self.header is not None:
            hdulist[0].header.update(self.header.to_header())  # primary = shared meta
        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )


class MapDatasetReaderWriter(ProductReaderWriter):
    PRODUCT = "MapDataset"


class SpectrumDatasetReaderWriter(ProductReaderWriter):
    PRODUCT = "SpectrumDataset"
