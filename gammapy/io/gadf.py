# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""GADF-specific metadata: column schemas, header models, and reader/writers.

Imports the format-agnostic machinery from ``core`` and registers
GADF into the shared registries (``DATA_FORMATS_MODELS``, ``PRODUCT_MODELS``).
"""

import copy
import logging
from typing import ClassVar, Optional, Union

from pydantic import ValidationInfo, model_validator

from gammapy.data import GTI, FixedPointingInfo, ObservationMetaData
from gammapy.data.io import EventListReader
from gammapy.datasets import MapDataset, SpectrumDataset
from gammapy.irf import (
    Background2D,
    Background3D,
    EffectiveAreaTable2D,
    EnergyDependentMultiGaussPSF,
    EnergyDispersion2D,
    PSF3D,
    PSFKing,
    RadMax2D,
    IRFMap,
)
from gammapy.maps import Map


# Format-agnostic machinery (adjust the import path to where core lives).
from gammapy.io.core import (
    DATA_FORMATS_MODELS,
    DEFAULT_DATA_FORMAT,
    DEFAULT_STRICT_READ,
    DEFAULT_STRICT_WRITE,
    GeneralFields,
    FitsTimeFields,
    FitsTimeConvenienceFields,
    HDUFields,
    HDUHeader,
    HDUReaderWriter,
    HDUListReaderWriter,
    HDUResponseFields,
    PRODUCT_MODELS,
    _field_names,
)

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# DL3 event-level tables
# --------------------------------------------------------------------------
GADF_V02_EVENT_TABLE_DEFINITION = {
    "EVENT_ID": {"dtype": "int", "required": True, "unit": None},
    "TIME": {"dtype": "float", "required": True, "unit": "s"},
    "RA": {"dtype": "float", "required": True, "unit": "deg"},
    "DEC": {"dtype": "float", "required": True, "unit": "deg"},
    "ENERGY": {"dtype": "float", "required": True, "unit": "TeV"},
    "EVENT_TYPE": {"dtype": "int8"},
    "MULTIP": {"dtype": "int"},
    "GLON": {"dtype": "float", "unit": "deg"},
    "GLAT": {"dtype": "float", "unit": "deg"},
    "ALT": {"dtype": "float", "unit": "deg"},
    "AZ": {"dtype": "float", "unit": "deg"},
    "DETX": {"dtype": "float", "unit": "deg"},
    "DETY": {"dtype": "float", "unit": "deg"},
    "THETA": {"dtype": "float", "unit": "deg"},
    "PHI": {"dtype": "float", "unit": "deg"},
    "GAMMANESS": {"dtype": "float"},
    "DIR_ERR": {"dtype": "float", "unit": "deg"},
    "ENERGY_ERR": {"dtype": "float", "unit": "TeV"},
    "COREX": {"dtype": "float", "unit": "m"},
    "COREY": {"dtype": "float", "unit": "m"},
    "CORE_ERR": {"dtype": "float", "unit": "m"},
    "XMAX": {"dtype": "float", "unit": "m"},
    "XMAX_ERR": {"dtype": "float", "unit": "m"},
    "HIL_MSW": {"dtype": "float", "unit": ""},
    "HIL_MSW_ERR": {"dtype": "float", "unit": ""},
    "HIL_MSL": {"dtype": "float", "unit": ""},
    "HIL_MSL_ERR": {"dtype": "float", "unit": ""},
}

GADF_V02_GTI_TABLE_DEFINITION = {
    "START": {"dtype": "float", "required": True, "unit": "s"},
    "STOP": {"dtype": "float", "required": True, "unit": "s"},
}

GADF_V02_POINTING_TABLE_DEFINITION = {
    "TIME": {"dtype": "float", "required": True, "unit": "s"},
    "RA_PNT": {"dtype": "float", "required": True, "unit": "deg"},
    "DEC_PNT": {"dtype": "float", "required": True, "unit": "deg"},
    "ALT_PNT": {"dtype": "float", "unit": "deg"},
    "AZ_PNT": {"dtype": "float", "unit": "deg"},
}

# --------------------------------------------------------------------------
# DL3 IRF tables
# --------------------------------------------------------------------------
GADF_V02_AEFF_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "EFFAREA": {"dtype": "float", "required": True, "unit": "m2", "ndim": 2},
}

GADF_V02_EDISP_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "MIGRA_LO": {"dtype": "float", "required": True, "unit": "", "ndim": 1},
    "MIGRA_HI": {"dtype": "float", "required": True, "unit": "", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "MATRIX": {"dtype": "float", "required": True, "unit": "", "ndim": 3},
}

GADF_V02_PSF_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "RAD_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "RAD_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "RPSF": {"dtype": "float", "required": True, "unit": "sr-1", "ndim": 3},
}

GADF_V02_PSF_3GAUSS_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "SCALE": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
    "SIGMA_1": {"dtype": "float", "required": True, "unit": "deg", "ndim": 2},
    "SIGMA_2": {"dtype": "float", "required": True, "unit": "deg", "ndim": 2},
    "SIGMA_3": {"dtype": "float", "required": True, "unit": "deg", "ndim": 2},
    "AMPL_2": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
    "AMPL_3": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
}

GADF_V02_PSF_KING_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "GAMMA": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
    "SIGMA": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
}

GADF_V02_PSF_TABLE_DEFINITION = {
    "ENERGY": {"dtype": "float", "required": True, "unit": "MeV", "ndim": 1},
    "EXPOSURE": {"dtype": "float", "required": True, "unit": "cm2 s", "ndim": 1},
    "PSF": {"dtype": "float", "required": True, "unit": "", "ndim": 2},
}

GADF_V02_THETA_TABLE_DEFINITION = {
    "THETA": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
}

GADF_V02_BKG_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "BKG": {"dtype": "float", "required": True, "unit": "s-1 MeV-1 sr-1", "ndim": 2},
}

GADF_V02_BKG_3D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "DETX_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "DETX_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "DETY_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "DETY_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "BKG": {"dtype": "float", "required": True, "unit": "s-1 MeV-1 sr-1", "ndim": 3},
}

GADF_V02_RAD_MAX_2D_TABLE_DEFINITION = {
    "ENERG_LO": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "ENERG_HI": {"dtype": "float", "required": True, "unit": "TeV", "ndim": 1},
    "THETA_LO": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "THETA_HI": {"dtype": "float", "required": True, "unit": "deg", "ndim": 1},
    "RAD_MAX": {"dtype": "float", "required": True, "unit": "deg", "ndim": 2},
}

# --------------------------------------------------------------------------
# DL4 Map tables
# --------------------------------------------------------------------------
GADF_V02_BANDS_TABLE_DEFINITION = {
    "CHANNEL": {"dtype": "int", "required": True, "unit": "", "ndim": 1},
    "E_MIN": {"dtype": "float", "required": False, "unit": "keV", "ndim": 1},
    "E_MAX": {"dtype": "float", "required": False, "unit": "keV", "ndim": 1},
    "ENERGY": {"dtype": "float", "required": False, "unit": "keV", "ndim": 1},
    "EVENT_TYPE": {"dtype": "int", "required": False, "unit": "", "ndim": 1},
}

GADF_V02_WCS_TABLE_DEFINITION = {
    "NPIX": {"dtype": "int", "required": False, "unit": "", "ndim": 2},
    "CRPIX": {"dtype": "float", "required": False, "unit": "deg", "ndim": 2},
    "CDELT": {"dtype": "float", "required": False, "unit": "deg", "ndim": 2},
}

GADF_V02_HPX_TABLE_DEFINITION = {
    "NSIDE": {"dtype": "int", "required": True, "unit": "", "ndim": 1},
}

# --------------- IRF DL3 HDU SPECIFICATION ---------------
# The key is the class tag.
GADF_IRF_DL3_HDU_SPECIFICATION = {
    "bkg_3d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "BKG_3D",
            "FOVALIGN": "RADEC",
        },
    },
    "bkg_2d": {
        "extname": "BACKGROUND",
        "column_name": "BKG",
        "mandatory_keywords": {
            "HDUCLAS2": "BKG",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "BKG_2D",
        },
    },
    "edisp_2d": {
        "extname": "ENERGY DISPERSION",
        "column_name": "MATRIX",
        "mandatory_keywords": {
            "HDUCLAS2": "EDISP",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "EDISP_2D",
        },
    },
    "psf_table": {
        "extname": "PSF_2D_TABLE",
        "column_name": "RPSF",
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
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
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "PSF_3GAUSS",
        },
    },
    "psf_king": {
        "extname": "PSF_2D_KING",
        "column_name": {"sigma": "SIGMA", "gamma": "GAMMA"},
        "mandatory_keywords": {
            "HDUCLAS2": "RPSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "PSF_KING",
        },
    },
    "psf_gtpsf": {
        "extname": "PSF_GTPSF",
        "mandatory_keywords": {
            "HDUCLAS2": "PSF",
            "HDUCLAS3": "FULL-ENCLOSURE",
            "HDUCLAS4": "GTPSF",
        },
    },
    "aeff_2d": {
        "extname": "EFFECTIVE AREA",
        "column_name": "EFFAREA",
        "mandatory_keywords": {
            "HDUCLAS2": "EFF_AREA",
            "HDUCLAS3": "FULL-ENCLOSURE",
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


GADF_EXTNAME_TO_TAG = {
    spec["extname"]: tag
    for tag, spec in GADF_IRF_DL3_HDU_SPECIFICATION.items()
    if "extname" in spec
}


def resolve_irf_tag_from_meta(meta):
    hduclas4 = meta.get("HDUCLAS4")
    if hduclas4 and hduclas4.lower() in GADF_IRF_DL3_HDU_SPECIFICATION:
        return hduclas4.lower()
    return GADF_EXTNAME_TO_TAG.get(meta.get("EXTNAME"))


def gadf_header_key(meta):
    key = meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    # IRF recoverable by EXTNAME -> RESPONSE, so the specific response header is used
    if key is None or key not in ("EVENTS", "GTI", "POINTING", "RESPONSE"):
        if resolve_irf_tag_from_meta(meta):
            return "RESPONSE"
    return key


def gadf_hdu_class_key(meta):
    from gammapy.irf.io import _get_hdu_type_and_class  # lazy

    key = meta.get("HDUCLAS4") or meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    if key in PRODUCT_MODELS:
        return key.upper()
    tag = None
    if not meta.get("HDUCLAS4"):  # only guess from EXTNAME if HDUCLAS4 missing
        tag = GADF_EXTNAME_TO_TAG.get(meta.get("EXTNAME"))
    if tag:
        return GADF_IRF_DL3_HDU_SPECIFICATION[tag]["mandatory_keywords"][
            "HDUCLAS4"
        ].upper()

    hdu_class = meta.get("HDUCLAS2", "unknown_clas")
    if hdu_class in ["EFF_AREA", "RPSF", "EDISP", "BKG"] and meta.get("HDUCLAS4"):
        _, hdu_class = _get_hdu_type_and_class(meta)  # CTA-1DC -> bkg_3d
    return hdu_class.upper()

    # # EXTNAME -> tag fallback (HDUCLAS4 absent)
    # tag = resolve_irf_tag_from_meta(meta)
    # if tag and tag in GADF_IRF_DL3_HDU_SPECIFICATION:
    #     return GADF_IRF_DL3_HDU_SPECIFICATION[tag]["mandatory_keywords"]["HDUCLAS4"].upper()

    # # CTA-1DC workaround: HDUCLAS2 is a known IRF type but HDUCLAS4 is non-standard
    # hdu_class = meta.get("HDUCLAS2", "unknown_clas")
    # if hdu_class in ["EFF_AREA", "RPSF", "EDISP", "BKG"] and meta.get("HDUCLAS4"):
    #     _, hdu_class = _get_hdu_type_and_class(meta)
    # return hdu_class.upper()


GADF_IRF_MAP_HDU_SPECIFICATION = {
    "edisp_kernel_map": "edisp",
    "edisp_map": "edisp",
    "psf_map": "psf",
    "psf_map_reco": "psf",
}

# --------------- VERSION / DOC REGISTRIES ---------------
GADF_DEFAULT_VERSION = "0.3"

_GADF_V02_TABLE = {
    "EVENTS": GADF_V02_EVENT_TABLE_DEFINITION,
    "GTI": GADF_V02_GTI_TABLE_DEFINITION,
    "POINTING": GADF_V02_POINTING_TABLE_DEFINITION,
    "AEFF_2D": GADF_V02_AEFF_2D_TABLE_DEFINITION,
    "EDISP_2D": GADF_V02_EDISP_2D_TABLE_DEFINITION,
    "PSF_TABLE": GADF_V02_PSF_2D_TABLE_DEFINITION,
    "PSF_3GAUSS": GADF_V02_PSF_3GAUSS_TABLE_DEFINITION,
    "PSF_KING": GADF_V02_PSF_KING_TABLE_DEFINITION,
    "PSF": GADF_V02_PSF_TABLE_DEFINITION,
    "THETA": GADF_V02_THETA_TABLE_DEFINITION,
    "BKG_2D": GADF_V02_BKG_2D_TABLE_DEFINITION,
    "BKG_3D": GADF_V02_BKG_3D_TABLE_DEFINITION,
    "RAD_MAX_2D": GADF_V02_RAD_MAX_2D_TABLE_DEFINITION,
    "BANDS": GADF_V02_BANDS_TABLE_DEFINITION,
    "IMAGE": GADF_V02_WCS_TABLE_DEFINITION,
    "SKYMAP": GADF_V02_HPX_TABLE_DEFINITION,
}

_GADF_V03_TABLE = copy.deepcopy(_GADF_V02_TABLE)
_GADF_V03_TABLE["PSF_3GAUSS"]["SCALE"]["unit"] = "sr-1"

# Nested registry: version -> HDU -> column-definition YAML.
GADF_PRODUCTS_TABLE_DEFINITION = {
    "0.2": _GADF_V02_TABLE,
    "0.3": _GADF_V03_TABLE,
}

GADF_HDUDOC = {
    "0.2": "https://github.com/open-gamma-ray-astro/gamma-astro-data-formats",
    "0.3": "https://gamma-astro-data-formats.readthedocs.io/en/v0.3/index.html",
}

# =====================================================================
# GADF header REQUIRED keywords, version-keyed.
# Registry contains only version-dependent keywords and rules
# =====================================================================

# Common non-pointing EVENTS keywords, identical across v0.2 and v0.3.
_HDU_COMMON = {
    "HDUCLASS",
    "HDUDOC",
    "HDUVERS",
    "HDUCLAS1",
}

_EVENTS_COMMON = _HDU_COMMON | {
    "HDUCLASS",
    "HDUDOC",
    "HDUVERS",
    "HDUCLAS1",
    "OBS_ID",
    "TSTART",
    "TSTOP",
    "ONTIME",
    "LIVETIME",
    "DEADC",
    "EQUINOX",
    "RADECSYS",
    "ORIGIN",
    "TELESCOP",
    "INSTRUME",
    "CREATOR",
}

_IRF_COMMON = _HDU_COMMON | {"HDUCLAS2", "HDUCLAS3", "HDUCLAS4"}

GADF_HEADER_REQUIRED = {
    "0.2": {
        # v0.2: RA_PNT/DEC_PNT unconditionally mandatory; OBS_MODE NOT required;
        #       no ALT_PNT/AZ_PNT, no DRIFT support.
        "EVENTS": _EVENTS_COMMON | {"RA_PNT", "DEC_PNT"},
        "RESPONSE": _IRF_COMMON,
    },
    "0.3": {
        # v0.3: OBS_MODE required; RA_PNT/DEC_PNT and ALT_PNT/AZ_PNT are conditional
        #       (see GADF_HEADER_CONDITIONAL_REQUIRED), so NOT in the static set.
        "EVENTS": _EVENTS_COMMON | {"OBS_MODE"},
        "RESPONSE": _IRF_COMMON,
    },
}

GADF_HEADER_CONDITIONAL_REQUIRED = {
    "0.2": {
        "EVENTS": [],  # no drift conditional in v0.2
    },
    "0.3": {
        "EVENTS": [
            # OBS_MODE=DRIFT -> ALT_PNT/AZ_PNT required, RA_PNT/DEC_PNT not;
            # otherwise      -> RA_PNT/DEC_PNT required, ALT_PNT/AZ_PNT not.
            ("OBS_MODE", "DRIFT", ["ALT_PNT", "AZ_PNT"], ["RA_PNT", "DEC_PNT"]),
        ],
    },
}


# --------------- GADF MANDATORY BASE FIELDS ---------------
from pydantic import BaseModel  # noqa: E402  (grouped with GADF field mixins)


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
    TSTART: Optional[float] = None
    TSTOP: Optional[float] = None
    ONTIME: Optional[float] = None
    LIVETIME: Optional[float] = None
    DEADC: Optional[float] = None
    EQUINOX: Optional[Union[float, str]] = None
    RADECSYS: Optional[str] = None


# --------------- GADF BASE HEADER ---------------
class GADFHDUHeader(HDUFields, HDUHeader):
    HDUCLASS: Optional[str] = "GADF"
    HDUVERS: Optional[str] = GADF_DEFAULT_VERSION

    def _tag_from_extname(self, extras=None):
        extname = self._resolve("EXTNAME", extras)
        for tag, spec in GADF_IRF_DL3_HDU_SPECIFICATION.items():
            if spec.get("extname") == extname:
                return tag
        return None

    def _irf_tag(self, extras=None):
        hduclas4 = self._resolve("HDUCLAS4", extras)
        return hduclas4.lower() if hduclas4 else self._tag_from_extname(extras)

    def _resolve_key(self, extras=None):
        """RESPONSE if HDUCLAS1 says so, or if an IRF tag is identifiable by EXTNAME."""
        key = self._resolve("HDUCLAS1", extras)
        if key:
            return key
        if self._tag_from_extname(extras) is not None:
            return "RESPONSE"
        return key

    def _check_extra_required(self, version, extras=None):
        """Tag-specific IRF keywords.

        The common IRF keywords come from HEADER_REQUIRED["RESPONSE"] via
        _resolve_key; here we only add the per-tag mandatory keywords.
        """
        tag = self._irf_tag(extras)
        if tag is None:
            return []
        extra_fields = GADF_IRF_TAG_REQUIRED_FIELDS.get(tag, {})
        return [k for k in extra_fields if not self._present_in_file(k, extras)]

    @model_validator(mode="after")
    def _enforce_gadf(self, info: ValidationInfo):
        strict = (info.context or {}).get("strict", False)
        hduvers = self._resolve("HDUVERS", (info.context or {}).get("extras", {}))
        if hduvers is not None and hduvers not in GADF_HDUDOC:
            msg = f"Non-standard HDUVERS {hduvers!r}; known: {list(GADF_HDUDOC)}"
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return self


# --------------- GADF EVENTS HEADER ---------------


class GADFEventsHeader(
    GADFHDUHeader,
    GeneralFields,
    GADFObsFields,
    FitsTimeFields,
    FitsTimeConvenienceFields,
    GADFEarthLocationFields,
    GADFPointingFields,
):
    HDUCLAS1: Optional[str] = "EVENTS"

    REQUIRED: ClassVar[frozenset] = _field_names(
        HDUFields,
        GeneralFields,
        GADFObsFields,
        FitsTimeFields,
        GADFEarthLocationFields,
        GADFPointingFields,
    )

    OBJECT: Optional[str] = None
    RA_OBJ: Optional[float] = None
    DEC_OBJ: Optional[float] = None
    OBSERVER: Optional[str] = None
    EV_CLASS: Optional[Union[int, str]] = None
    TELAPSE: Optional[float] = None
    TASSIGN: Optional[str] = None
    TELLIST: Optional[Union[str, list]] = None
    N_TELS: Optional[int] = None
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


# --------------- GADF GTI / POINTING HEADERS ---------------
class GADFGTIHeader(GADFHDUHeader, FitsTimeFields, FitsTimeConvenienceFields):
    HDUCLAS1: Optional[str] = "GTI"
    REQUIRED: ClassVar[frozenset] = _field_names(FitsTimeFields)


class GADFPointingHeader(GADFHDUHeader, FitsTimeFields, FitsTimeConvenienceFields):
    HDUCLAS1: Optional[str] = "POINTING"
    REQUIRED: ClassVar[frozenset] = _field_names(FitsTimeFields)


# --------------- GADF IRF HEADER ---------------
GADF_IRF_TAG_REQUIRED_FIELDS = {
    "bkg_2d": ["FOVALIGN"],
    "bkg_3d": ["FOVALIGN"],
}


class GADFResponseHeader(GADFHDUHeader, HDUResponseFields, GeneralFields):
    HDUCLAS1: Optional[str] = "RESPONSE"
    EXTNAME: Optional[str] = None
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, HDUResponseFields) | {
        "EXTNAME"
    }

    OBS_ID: Optional[int] = None
    LO_THRES: Optional[float] = None
    HI_THRES: Optional[float] = None
    FOVALIGN: Optional[str] = None

    @classmethod
    def from_tag(cls, tag, **overrides):
        if tag not in GADF_IRF_DL3_HDU_SPECIFICATION:
            raise ValueError(
                f"Unknown IRF tag {tag!r}; known: {list(GADF_IRF_DL3_HDU_SPECIFICATION)}"
            )
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        kwargs = {"EXTNAME": spec["extname"], **spec["mandatory_keywords"], **overrides}
        return cls.from_meta(kwargs)

    def apply_tag(self, tag):
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        self.EXTNAME = spec["extname"]
        for key, value in spec["mandatory_keywords"].items():
            if key in type(self).model_fields:
                setattr(self, key, value)
        return self


# --------------- GADF DL4 HEADERS ---------------
class GADFBandsHeader(GADFHDUHeader):
    HDUCLAS1: Optional[str] = "BANDS"
    REQUIRED: ClassVar[frozenset] = frozenset()


class GADFWcsSkymapHeader(GADFHDUHeader):
    HDUCLAS1: Optional[str] = "IMAGE"
    BANDSHDU: Optional[str] = None  # name of the companion BANDS HDU
    REQUIRED: ClassVar[frozenset] = frozenset()  # WCS map compliance = valid WCS cards


class GADFHpxSkymapHeader(GADFHDUHeader):
    HDUCLAS1: Optional[str] = "SKYMAP"
    PIXTYPE: Optional[str] = "HEALPIX"
    ORDERING: Optional[str] = None  # NESTED | RING
    INDXSCHM: Optional[str] = None  # IMPLICIT | EXPLICIT | SPARSE (default IMPLICIT)
    ORDER: Optional[int] = None  # log2(NSIDE) or -1
    NSIDE: Optional[int] = None  # superseded by NSIDE column if BANDS defined
    COORDSYS: Optional[str] = None  # CEL | GAL
    BANDSHDU: Optional[str] = None
    REQUIRED: ClassVar[frozenset] = frozenset({"PIXTYPE"})  # per spec, must be HEALPIX


GADF_HEADER_MODELS = {
    "BASE": GADFHDUHeader,
    "EVENTS": GADFEventsHeader,
    "GTI": GADFGTIHeader,
    "POINTING": GADFPointingHeader,
    "RESPONSE": GADFResponseHeader,
    "BANDS": GADFBandsHeader,
    "IMAGE": GADFWcsSkymapHeader,
    "SKYMAP": GADFHpxSkymapHeader,
}


# --------------- GADF READER/WRITERS ---------------


class GADFHDUReaderWriter(HDUReaderWriter):
    """IO class specialised to the GADF HDU."""

    DEFAULT_HDU = None
    GADF_HEADER_MODEL = None
    format = "GADF"


class GADFEventsReaderWriter(HDUReaderWriter):
    """IO class specialised to the EVENTS HDU."""

    DEFAULT_HDU = "EVENTS"
    GADF_HEADER_MODEL = GADFEventsHeader
    format = "GADF"

    @classmethod
    def from_eventlist(
        cls, eventlist, version=GADF_DEFAULT_VERSION, strict=DEFAULT_STRICT_READ
    ):
        table = eventlist.table
        header = cls.GADF_HEADER_MODEL.from_meta(table.meta, strict)
        cls.format_validator(header, table, cls.format, version, strict)
        return cls(
            table=table, header=header, data=None, hdu=cls.DEFAULT_HDU, version=version
        )

    def to_pointing(self):
        """Build FixedPointingInfo."""
        return FixedPointingInfo.from_fits_header(self.header.to_meta())


class GADFGTIReaderWriter(HDUReaderWriter):
    """IO class specialised to the GTI HDU."""

    DEFAULT_HDU = "GTI"
    GADF_HEADER_MODEL = GADFGTIHeader
    format = "GADF"


def resolve_irf_tag(meta):
    """Identify the IRF tag from a RESPONSE HDU header via HDUCLAS4."""
    hduclas4 = meta.get("HDUCLAS4")
    if hduclas4 is None:
        raise ValueError("Missing HDUCLAS4: cannot identify the IRF type.")
    tag = hduclas4.lower()
    if tag not in GADF_IRF_DL3_HDU_SPECIFICATION:
        raise ValueError(
            f"Unknown IRF HDUCLAS4={hduclas4!r}; "
            f"known: {list(GADF_IRF_DL3_HDU_SPECIFICATION)}"
        )
    return tag


class GADFResponseReaderWriter(HDUReaderWriter):
    """IO for GADF RESPONSE HDUs.

    Header/metadata is handled here. The binned data (axes + payload) is
    delegated to the gammapy IRF object's own from_table / to_table.
    """

    GADF_HEADER_MODEL = GADFResponseHeader
    format = "GADF"

    @classmethod
    def from_irf(cls, irf, version=GADF_DEFAULT_VERSION, strict=DEFAULT_STRICT_READ):
        table = irf.to_table()
        tag = irf.tag if isinstance(irf.tag, str) else irf.tag[0]
        if strict:
            cls.table_validator(tag.upper(), version=version).run(table)
        spec = GADF_IRF_DL3_HDU_SPECIFICATION[tag]
        meta = {**dict(getattr(irf, "meta", {}) or {}), **spec["mandatory_keywords"]}
        header = cls.GADF_HEADER_MODEL.from_meta(meta, strict)
        return cls(
            table=table,
            header=header,
            data=None,
            hdu=gadf_hdu_class_key(meta),
            format=cls.format,
            version=version,
        )


class GADFPointingReaderWriter(HDUReaderWriter):
    """IO for GADF POINTING HDU.

    Gammapy only supports the class FixedPointingInfo and assumes it by default. Fixed pointing information is extracted from the EVENTS HDU."""

    DEFAULT_HDU = "POINTING"
    GADF_HEADER_MODEL = GADFPointingHeader
    format = "GADF"

    @classmethod
    def from_pointing(
        cls, fpi, version=GADF_DEFAULT_VERSION, strict=DEFAULT_STRICT_READ
    ):
        header_dict = dict(
            fpi.to_fits_header(format=cls.format.lower(), version=version)
        )
        header = cls.GADF_HEADER_MODEL.from_meta(
            header_dict, strict=strict, version=version
        )
        return cls(
            table=None,
            data=None,
            header=header,
            hdu=cls.DEFAULT_HDU,
            format=cls.format.lower(),
            version=version,
        )


GADF_READER_WRITER_MODELS = {
    "EVENTS": GADFEventsReaderWriter,
    "GTI": GADFGTIReaderWriter,
    "RESPONSE": GADFResponseReaderWriter,
    "POINTING": GADFPointingReaderWriter,
}


# --------------- REGISTRY INJECTION ---------------
# Populate the shared registries owned by core.
DATA_FORMATS_MODELS["GADF"] = {
    "TABLE": GADF_PRODUCTS_TABLE_DEFINITION,
    "HEADER": GADF_HEADER_MODELS,
    "HEADER_REQUIRED": GADF_HEADER_REQUIRED,  # version -> hdu -> set
    "HEADER_CONDITIONAL_REQUIRED": GADF_HEADER_CONDITIONAL_REQUIRED,  # version -> hdu -> rules
    "READER_WRITER": GADF_READER_WRITER_MODELS,
    "HEADER_KEY": gadf_header_key,
    "HDU_CLASS_KEY": gadf_hdu_class_key,
}

PRODUCT_MODELS.update(
    {
        "EVENTS": EventListReader,
        "GTI": GTI,
        "POINTING": FixedPointingInfo,
        "AEFF_2D": EffectiveAreaTable2D,
        "EDISP_2D": EnergyDispersion2D,
        "PSF_TABLE": PSF3D,
        "PSF_3GAUSS": EnergyDependentMultiGaussPSF,
        "PSF_KING": PSFKing,
        "BKG_2D": Background2D,
        "BKG_3D": Background3D,
        "RAD_MAX_2D": RadMax2D,
        "AEFF_BANDS": IRFMap,
        "PSF_BANDS": IRFMap,
        "EDISP_BANDS": IRFMap,
        "PSF_MAP": IRFMap,
        # "IMAGE": Map,
        # "BANDS": Map,
        # "SKYMAP": Map,
        "observation_metadata": ObservationMetaData,
        "map": Map,
        "MapDataset": MapDataset,
        "SpectrumDataset": SpectrumDataset,
    }
)


class GADFHDUListReaderWriter(HDUListReaderWriter):
    """GADF-aware file reader: framework validation + gammapy Map reconstruction."""

    @classmethod
    def read(cls, filename, *args, **kwargs):
        obj = super().read(filename, *args, **kwargs)
        obj._filename = filename
        return obj

    def _skymap_keys(self):
        keys = []
        for key, rw in self.hdu_dict.items():
            if key == "PRIMARY":
                continue
            hdr = rw.header.to_meta()
            if (
                hdr.get("PIXTYPE") == "HEALPIX"
                or hdr.get("HDUCLAS1") in ("SKYMAP", "IMAGE")
                or "CTYPE1" in hdr
            ):
                keys.append(key)
        if len(keys) == 0:
            extname = self.hdu_dict["PRIMARY"].header.to_meta().get("EXTNAME", None)
            if extname:
                keys.append(extname)
        return keys

    def to_map(self, hdu=None, hdu_class=None):
        from gammapy.maps import Map

        if getattr(self, "_filename", None) is None:
            raise ValueError("to_map requires the source filename (read from file).")
        if hdu is None:
            keys = self._skymap_keys()
            if len(keys) != 1:
                raise ValueError(f"Specify hdu; SKYMAP HDUs found: {keys}")
            hdu = keys[0]
        try:
            from gammapy.irf import IRF_REGISTRY

            cls = IRF_REGISTRY.get_cls(hdu_class)
            return cls.read(self._filename, hdu=hdu.lower())
        except Exception as e:
            log.warning(
                "IRF reconstruction failed for hdu=%r, hdu_class=%r (%s: %s); "
                "falling back to Map.read.",
                hdu,
                hdu_class,
                type(e).__name__,
                e,
            )
        return Map.read(self._filename, hdu=hdu)

    def to_maps(self):
        return {key: self.to_map(hdu=key) for key in self._skymap_keys()}
