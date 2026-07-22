# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Format-agnostic metadata machinery.

This module holds the parts of the serialization framework that know nothing
about a specific data format (GADF, OGIP, ...): the base Pydantic field
mixins, the base header models, the reader/writer classes, and *empty*
registries that concrete format modules populate at import time.

Dependency direction is one-way: format modules (e.g. ``gadf``)
import from here and register into the registries below. This module must not
import any format-specific or gammapy code, to avoid circular imports.
"""

import logging
from typing import ClassVar, Literal, Optional, Union

from astropy.io import fits
from astropy.table import Table
from astropy.units import UnitTypeError
from pydantic import Field, BaseModel, ConfigDict, ValidationInfo, model_validator

from gammapy.utils.scripts import make_path
from gammapy.utils.table_validator import TableValidator

log = logging.getLogger(__name__)


# --------------- INJECTABLE REGISTRIES ---------------
# Concrete format modules fill these at import time, e.g.
#   DATA_FORMATS_MODELS["GADF"] = {"TABLE": ..., "HEADER": ..., "READER_WRITER": ...}
#   PRODUCT_MODELS.update({...})
# Core logic reads them at call time, by which point they are populated.
DATA_FORMATS_MODELS: dict = {}
PRODUCT_MODELS: dict = {}

DEFAULT_DATA_FORMAT = "GADF"
DEFAULT_DATA_FORMAT_VERSION = "0.3"
DEFAULT_STRICT_READ = False
DEFAULT_STRICT_WRITE = True


# --------------- FORMAT-AGNOSTIC BASE FIELDS ---------------
# Also usable by other formats (e.g. OGIP). Fields are declared optional but
# may be required at validation (enforced via the header REQUIRED set).


class FitsTimeFields(BaseModel):
    MJDREFI: Optional[int] = None
    MJDREFF: Optional[float] = None
    TIMEUNIT: Optional[str] = None
    TIMESYS: Optional[str] = None
    TIMEREF: Optional[str] = None


class FitsTimeConvenienceFields(BaseModel):
    DATE_OBS: Optional[str] = Field(None, alias="DATE-OBS")
    DATE_BEG: Optional[str] = Field(None, alias="DATE-BEG")
    DATE_AVG: Optional[str] = Field(None, alias="DATE-AVG")
    TIME_OBS: Optional[str] = Field(None, alias="TIME-OBS")
    DATE_END: Optional[str] = Field(None, alias="DATE-END")
    TIME_END: Optional[str] = Field(None, alias="TIME-END")


class GeneralFields(BaseModel):
    TELESCOP: Optional[str] = None
    INSTRUME: Optional[str] = None
    ORIGIN: Optional[str] = None
    CREATOR: Optional[str] = None


class HDUFields(BaseModel):
    # Neutral defaults: concrete formats set HDUCLASS/HDUVERS themselves.
    HDUCLASS: Optional[str] = None
    HDUVERS: Optional[str] = None
    HDUDOC: Optional[str] = None
    HDUCLAS1: Optional[str] = None


class HDUResponseFields(BaseModel):
    HDUCLAS2: Optional[str] = None
    HDUCLAS3: Optional[str] = None
    HDUCLAS4: Optional[str] = None


def _field_names(*models):
    return frozenset(name for m in models for name in m.model_fields)


# --------------- BASE HEADERS ---------------
class HDUHeader(BaseModel):
    """Base HDU header model (lenient by default, strict via context)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")
    HDUCLASS: Optional[str] = None
    comments: Optional[Union[str, list]] = None
    history: Optional[Union[str, list]] = None
    REQUIRED: ClassVar[frozenset] = frozenset()

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name)

    @classmethod
    def from_meta(cls, header, strict=False, version=None):
        declared = {cls.model_fields[k].alias or k for k in cls.model_fields}
        kwargs = {k: header[k] for k in declared if k in header}
        extras = {k: header[k] for k in set(dict(header)) - declared}
        obj = cls.model_validate(
            kwargs, context={"strict": strict, "version": version, "extras": extras}
        )
        object.__setattr__(obj, "_extras", extras)
        return obj

    def to_meta(self):
        meta = self.model_dump(by_alias=True, exclude_none=True)
        meta.update(getattr(self, "_extras", {}))
        return meta

    def _format_models(self):
        return DATA_FORMATS_MODELS.get(getattr(self, "HDUCLASS", None))

    def _resolve(self, name, extras=None):
        val = getattr(self, name, None)
        if val in (None, ""):
            extras = extras if extras is not None else getattr(self, "_extras", {})
            alias = self.model_fields[name].alias if name in self.model_fields else None
            val = extras.get(name)
            if val in (None, "") and alias:
                val = extras.get(alias)
        return val if val not in (None, "") else None

    def _present_in_file(self, name, extras=None):
        extras = extras if extras is not None else getattr(self, "_extras", {})
        val = getattr(self, name, None)
        if name in self.model_fields_set and val not in (None, "", []):
            return True
        alias = self.model_fields[name].alias if name in self.model_fields else None
        for key in (name, alias):
            if key and extras.get(key) not in (None, "", []):
                return True
        return False

    def _resolve_key(self, extras=None):
        """HEADER_REQUIRED lookup key. Overridable by format subclasses."""
        return self._resolve("HDUCLAS1", extras)

    def _check_extra_required(self, version, extras=None):
        """Format-specific extra requirements. Overridden by subclasses."""
        return []

    def _check_required(self, version, extras=None):
        models = self._format_models()
        key = self._resolve_key(extras)
        required = (
            models["HEADER_REQUIRED"].get(version, {}).get(key) if models else None
        )
        if models and "HEADER_REQUIRED" in models:
            required = models["HEADER_REQUIRED"].get(version, {}).get(key)
        if required is None:
            required = self.REQUIRED
        missing = [
            (self.model_fields[name].alias or name)
            if name in self.model_fields
            else name
            for name in required
            if not self._present_in_file(name, extras)
        ]
        missing += self._check_extra_required(version, extras)
        return missing

    def _check_conditional(self, version, extras=None):
        models = self._format_models()
        if not models:
            return []
        key = self._resolve_key(extras)
        rules = (
            models.get("HEADER_CONDITIONAL_REQUIRED", {}).get(version, {}).get(key, [])
        )
        errors = []
        for field, value, req_if, req_if_not in rules:
            actual = self._resolve(field, extras)
            required = req_if if actual == value else req_if_not
            missing = [k for k in required if not self._present_in_file(k, extras)]
            if missing:
                errors.append(f"{missing} required when {field}={actual!r}")
        return errors

    @model_validator(mode="after")
    def _enforce(self, info: ValidationInfo):
        ctx = info.context or {}
        strict = ctx.get("strict", False)
        extras = ctx.get("extras", {})
        version = (
            ctx.get("version")
            or self._resolve("HDUVERS", extras)
            or DEFAULT_DATA_FORMAT_VERSION
        )
        errors = []
        missing = self._check_required(version, extras)
        if missing:
            errors.append(f"Missing mandatory keyword(s): {missing}")
        errors += self._check_conditional(version, extras)
        for msg in errors:
            if strict:
                raise ValueError(msg)
            log.warning(msg)
        return self


class PrimaryHDUHeader(HDUHeader, GeneralFields):
    """Header for the PRIMARY HDU: general/provenance metadata only.

    No HDUCLASn/HDUVERS format keywords (the primary carries no data class),
    and no REQUIRED enforcement (nothing is mandatory at the primary level yet).
    """

    HDUCLASS: Literal["PRIMARY"] = "PRIMARY"


class HDUProductHeader(HDUFields, HDUHeader, GeneralFields):
    HDUCLAS1: "Literal['PRODUCT']" = "PRODUCT"
    REQUIRED: ClassVar[frozenset] = _field_names(HDUFields, GeneralFields)


# class CustomHDUHeader(HDUHeader, HDUResponseFields, GeneralFields):
#     model_config = ConfigDict(extra="allow")

#     @classmethod
#     def from_header(cls, header):
#         return cls.model_construct(**dict(header))


# --------------- DISPATCH HELPERS ---------------
class UnknownHDUClass(IOError):
    """Raised when a file contains an unknown HDUCLASS."""


def _hdu_key(meta, fallback=None):
    """Helper function for the hdu dictionary keys."""
    hduclas2 = meta.get("HDUCLAS2")
    if hduclas2 is not None:
        return hduclas2.upper()
    return meta.get("EXTNAME") or fallback


def _header_key(meta):
    """Generic header-model dispatch key. Format modules may register an override."""
    key = meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    return key


def _hdu_class_key(meta):
    """Generic product/schema dispatch key. Format modules may register an override."""
    key = meta.get("HDUCLAS4") or meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if key and "BANDS" in key:
        return "BANDS"
    return key.upper() if key else "UNKNOWN"


def _check_data_format(meta, format, strict):
    meta_format = meta.get("HDUCLASS")
    if meta_format != format and strict:
        raise ValueError(
            f"HDUReaderWriter expected a {format} HDU class, {meta_format} was provided."
        )
    return meta_format


def apply_meta_to_header(fits_header, header):
    meta = header.to_meta()
    for meta_key in ["comments", "COMMENT", "history"]:
        if meta_key in meta.keys():
            meta.pop(meta_key)
    fits_header.update(meta)


# --------------- BASE READER/WRITER ---------------
class HDUReaderWriter:
    """IO for a single HDU, dispatched by (format, version, HDU)."""

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
            return TableValidator.from_dict(definition)
        log.warning("No table definition for %s", format)

    @classmethod
    def validate_header(cls, header, version, strict):
        if header is None or not hasattr(header, "_check_required"):
            return None
        version = version or DEFAULT_DATA_FORMAT_VERSION
        errors = header._check_required(version) + header._check_conditional(version)
        if errors and strict:
            raise ValueError("Header not compliant: " + "; ".join(errors))
        return not errors

    @classmethod
    def validate_table(cls, table, format, version, strict):
        """Enforce the format's column schema on a table."""
        if table is None:
            return None
        models = DATA_FORMATS_MODELS.get(format, {})
        class_key = models.get("HDU_CLASS_KEY", _hdu_class_key)
        hdu_class_key = class_key(table.meta)
        errors = []
        try:
            cls.table_validator(hdu=hdu_class_key, format=format, version=version).run(
                table
            )
        except (KeyError, TypeError, UnitTypeError, ValueError) as e:
            errors.append(f"table: {e}")
        if errors:
            msg = "Table not compliant: " + ", ".join(errors)
            if strict:
                raise ValueError(msg)
            log.warning("%s", msg)
        return not errors

    @classmethod
    def format_validator(cls, header, table, format, version, strict):
        header_valid = cls.validate_header(header, version, strict)
        table_valid = cls.validate_table(table, format, version, strict)
        return header_valid, table_valid

    @classmethod
    def _from_fits_hdu(
        cls, fits_hdu, format, version, strict, validate=True, verbose=True
    ):
        if isinstance(fits_hdu, fits.PrimaryHDU):
            header = PrimaryHDUHeader.from_meta(dict(fits_hdu.header))
            return cls(
                table=None,
                data=None,
                header=header,
                hdu=fits_hdu.name,
                format=None,
                version=None,
            )

        if fits_hdu.is_image:
            meta, data, table = dict(fits_hdu.header), fits_hdu.data, None
        else:
            table = Table.read(fits_hdu)
            meta, data = dict(table.meta), None

        # detect format first, so we can use the format's own key resolvers
        # workaround needed for cta-1dc data encoded with OGIP format
        telescope = meta.get("TELESCOP")
        instrument = meta.get("INSTRUME")
        is_cta_1dc = (telescope and telescope == "CTA") and (
            instrument and instrument == "1DC"
        )
        detected = "GADF" if is_cta_1dc else _check_data_format(meta, format, strict)

        meta_format = format if detected in (None, "UNKNOWN") else detected
        models = DATA_FORMATS_MODELS.get(meta_format)

        # key resolvers: format-specific if registered, else generic
        header_key = (models or {}).get("HEADER_KEY", _header_key)
        class_key = (models or {}).get("HDU_CLASS_KEY", _hdu_class_key)
        hkey = "IMAGE" if fits_hdu.is_image else header_key(meta)
        hdu_class = class_key(meta)

        log.info(
            "Reading %s HDU%s", hkey, f" ({hdu_class})" if hdu_class != hkey else ""
        )

        if models is None:
            if meta_format == format and format not in DATA_FORMATS_MODELS:
                raise ValueError(f"Requested format {format!r} is not registered.")
            if verbose:
                log.warning(
                    "Detected format %r not registered; neutral HDUHeader.", meta_format
                )
            rw = cls(
                table=table,
                data=data,
                header=HDUHeader.from_meta(meta),
                hdu=hdu_class,
                format=meta_format,
                version=None,
            )
            rw._header_valid = None
            rw._table_valid = None
            return rw

        BaseHeader = models["HEADER"].get("BASE")
        HeaderModel = models["HEADER"].get(hkey) or BaseHeader
        if HeaderModel is None:
            raise KeyError(
                f"No {meta_format} header model for HDU key {hkey!r} "
                f"and no BASE header registered."
            )

        header_constructed = True
        try:
            header = HeaderModel.from_meta(meta, strict=strict, version=version)
        except (ValueError, KeyError, TypeError, UnitTypeError) as e:
            if strict:
                raise
            header_constructed = False
            if verbose:
                log.warning(
                    "%s\nHeader construction failed, using %s BASE header.",
                    e,
                    meta_format,
                )
            try:
                header = BaseHeader.from_meta(meta, strict=False, version=version)
            except (ValueError, KeyError, TypeError, UnitTypeError) as e2:
                if verbose:
                    log.warning(
                        "%s\n%s BASE header failed too, using neutral header.",
                        e2,
                        meta_format,
                    )
                header = HDUHeader.from_meta(meta)

        header_valid = (
            cls.validate_header(header, version, strict) if validate else None
        )
        table_valid = (
            cls.validate_table(table, meta_format, version, strict)
            if validate and table is not None
            else None
        )

        ReaderWriter = models.get("READER_WRITER", {}).get(hkey, cls)
        rw = ReaderWriter(
            table=table,
            data=data,
            header=header,
            hdu=hdu_class,
            format=meta_format,
            version=None if not header_constructed else version,
        )
        rw._header_valid = header_valid
        rw._table_valid = table_valid
        return rw

    @classmethod
    def read(
        cls,
        filename,
        hdu=None,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
        validate=True,
        verbose=True,
    ):
        from gammapy.io import gadf  # noqa: F401  (import side effect: registry injection)

        hdu = hdu or cls.DEFAULT_HDU
        if hdu is None:
            raise ValueError("HDUReaderWriter objects require an `hdu`.")
        with fits.open(make_path(filename), checksum=checksum, memmap=False) as hdulist:
            return cls._from_fits_hdu(
                hdulist[hdu], format, version, strict, validate, verbose
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
        validate=True,
        verbose=True,
    ):
        if validate:
            header_valid, table_valid = self.format_validator(
                self.header, self.table, format, version, strict
            )
        else:
            header_valid, table_valid = None, None

        table_hdu = fits.BinTableHDU(self.table, name=self.hdu)
        apply_meta_to_header(table_hdu.header, self.header)
        if verbose:
            hstr = "OK" if header_valid else "FAIL"
            tstr = "OK" if table_valid else "FAIL"
            log.info(
                f"HDU: {self.hdu:<10}  TEST FMT: {format} VER: {version} HDR: {hstr} TAB: {tstr}"
            )
        return table_hdu

    def to_image_hdu(self):
        image_hdu = fits.ImageHDU(data=self.data, name=self.hdu)
        apply_meta_to_header(image_hdu.header, self.header)
        return image_hdu

    def _to_hdu(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
        validate=True,
        verbose=True,
    ):
        if self.table is not None:
            return self.to_table_hdu(format, version, strict, validate, verbose)
        if self.data is not None:
            return self.to_image_hdu()
        raise ValueError(f"{self.hdu}: neither table nor data set, nothing to write.")

    def to_hdulist(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
        validate=True,
        verbose=True,
    ):
        return [
            fits.PrimaryHDU(),
            self._to_hdu(format, version, strict, validate, verbose),
        ]

    def write(
        self,
        filename,
        overwrite=False,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
        validate=True,
        verbose=True,
    ):
        hdulist = fits.HDUList(
            self.to_hdulist(format, version, strict, validate, verbose)
        )
        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )


class HDUListReaderWriter:
    """IO for a whole multi-HDU file."""

    def __init__(self, hdu_dict):
        self.hdu_dict = hdu_dict

    @classmethod
    def read(
        cls,
        filename,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
        validate=True,
        verbose=True,
    ):
        from gammapy.io import gadf  # noqa: F401  (import side effect: registry injection)

        filename = make_path(filename)
        if verbose:
            log.info(
                "\n------------------------------------------------\n%s \nTESTED FMT: %s VER: %s (strict=%s)",
                filename,
                format,
                version,
                strict,
            )

        with fits.open(filename, checksum=checksum, memmap=False) as hdulist:
            hdu_dict = {}
            validation_info = []
            for hdu in hdulist:
                rw = HDUReaderWriter._from_fits_hdu(
                    hdu, format, version, strict, validate, verbose
                )
                key = (
                    "PRIMARY"
                    if isinstance(hdu, fits.PrimaryHDU)
                    else _hdu_key(dict(hdu.header), fallback=hdu.name)
                )
                if key in hdu_dict:
                    log.warning("Duplicate HDU key %r; keeping the first.", key)
                    continue
                hdu_dict[key] = rw

                if verbose:
                    fmt = rw.format or ""
                    ver = (
                        f"{getattr(rw.header, 'HDUVERS', '?')}"
                        if rw.format in DATA_FORMATS_MODELS
                        else ""
                    )
                    h = getattr(rw, "_header_valid", None)
                    t = getattr(rw, "_table_valid", None)
                    hstr = "" if h is None else ("OK" if h else "FAIL")
                    tstr = "" if t is None else ("OK" if t else "FAIL")
                    validation_info.append((key, fmt, ver, hstr, tstr))

            if verbose and validation_info:
                # Some formatting for human readable output
                kw = max(len(r[0]) for r in validation_info)
                fw = max(len(r[1]) for r in validation_info)
                vw = max(len(r[2]) for r in validation_info)
                hw = max(len(r[3]) for r in validation_info)
                header_row = f"\n{'HDU':<{kw}}  {'FMT':<{fw}} {'VER':<{vw}} {'HDR':<{hw}} {'TAB'}"
                validation_lines = [
                    f"{k:<{kw}}  {f:<{fw}} {v:<{vw}} {h:<{hw}} {t}".rstrip()
                    for k, f, v, h, t in validation_info[1:]
                ]
                log.info("%s\n%s", header_row, "\n".join(validation_lines))
        return cls(hdu_dict=hdu_dict)

    @property
    def format_list(self):
        return {key: rw.format for key, rw in self.hdu_dict.items()}

    def to_product_dict(self):
        clas_to_type = {"rpsf": "psf", "eff_area": "aeff"}

        product_dict = {}
        for hdu_name, rw in self.hdu_dict.items():
            if hdu_name == "PRIMARY":
                continue
            # Conversion to keys accepted by Observation
            hdu_type = clas_to_type.get(hdu_name.lower(), hdu_name.lower())
            product_dict[hdu_type] = rw.to_product()

        # pointing: only from EVENTS, only when no dedicated POINTING HDU exists
        events_rw = self.hdu_dict.get("EVENTS")
        if (
            events_rw is not None
            and "POINTING" not in self.hdu_dict
            and hasattr(events_rw, "to_pointing")
        ):
            product_dict["pointing"] = events_rw.to_pointing()

        return product_dict

    def to_hdulist(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
        verbose=True,
    ):
        hdulist = [fits.PrimaryHDU()]
        for hdu in self.hdu_dict:
            if hdu == "PRIMARY":
                continue
            hdulist.append(self.hdu_dict[hdu]._to_hdu(format, version, strict, verbose))
        return hdulist

    def write(
        self,
        filename,
        overwrite=False,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
        verbose=True,
    ):
        hdulist = fits.HDUList(self.to_hdulist(format, version, strict, verbose))
        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )


class ProductReaderWriter:
    """IO for gammapy products (DL3/DL4/DL5)."""

    PRODUCT = None

    def __init__(
        self,
        filename=None,
        hdu=None,
        checksum=False,
        product=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
    ):
        from gammapy.io import gadf  # noqa: F401  (import side effect: registry injection)

        _filename = filename
        hdu = hdu or self.PRODUCT
        if product is None:
            rw = HDUListReaderWriter.read(filename, checksum, format, version, strict)
            product = rw.hdu_dict[hdu].to_product()
            if product is None:
                raise ValueError(f"HDU {hdu!r} did not resolve to a gammapy product.")
            self.product = product
            self.header = rw.hdu_dict[hdu].header
            format = rw.hdu_dict[hdu].format
        else:
            self.product = product
            meta = product.meta
            hkey = _header_key(meta)
            self.header = DATA_FORMATS_MODELS[format]["HEADER"][hkey].from_meta(
                meta, version, strict
            )
        self.format = format
        self.version = version

    @classmethod
    def read(
        cls,
        filename,
        hdu=None,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
    ):
        hdu = hdu or cls.PRODUCT
        return cls(filename, hdu, checksum, None, format, version, strict).product

    def write(
        self,
        filename,
        overwrite=False,
        checksum=False,
        format=None,
        version=None,
        strict=DEFAULT_STRICT_WRITE,
    ):
        format = format or self.format
        version = version or self.version
        hdulist = self.product.to_hdulist()

        if self.header is not None:
            HDUReaderWriter.validate_header(self.header, version, strict)
        for hdu in hdulist:
            if isinstance(hdu, fits.PrimaryHDU) or hdu.is_image:
                continue
            HDUReaderWriter.validate_table(Table.read(hdu), format, version, strict)

        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )
