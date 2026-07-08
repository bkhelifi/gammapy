# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Format-agnostic metadata machinery.

This module holds the parts of the serialization framework that know nothing
about a specific data format (GADF, OGIP, ...): the base Pydantic field
mixins, the base header models, the reader/writer classes, and *empty*
registries that concrete format modules populate at import time.

Dependency direction is one-way: format modules (e.g. ``gadf_metadata``)
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
    HDUCLASS: Optional[str] = "CUSTOM"
    comments: Optional[Union[str, list]] = None
    REQUIRED: ClassVar[frozenset] = frozenset()

    def __getitem__(self, name):
        try:
            return getattr(self, name)
        except AttributeError:
            raise KeyError(name)

    @classmethod
    def from_header(cls, header, version=None, strict=False):
        declared = {cls.model_fields[k].alias or k for k in cls.model_fields}
        kwargs = {k: header[k] for k in declared if k in header}
        obj = cls.model_validate(kwargs, context={"strict": strict, "version": version})
        object.__setattr__(
            obj, "_extras", {k: header[k] for k in set(dict(header)) - declared}
        )
        return obj

    def to_header(self):
        hdr = self.model_dump(by_alias=True, exclude_none=True)
        hdr.update(getattr(self, "_extras", {}))
        return hdr

    def _format_models(self):
        return DATA_FORMATS_MODELS.get(getattr(self, "HDUCLASS", None))

    def _check_required(self, version):
        models = self._format_models()
        key = getattr(self, "HDUCLAS1", None)
        required = None
        if models and "HEADER_REQUIRED" in models:
            required = models["HEADER_REQUIRED"].get(version, {}).get(key)
        if required is None:
            required = self.REQUIRED
        return [
            (self.model_fields[name].alias or name)
            if name in self.model_fields
            else name
            for name in required
            if getattr(self, name, None) is None
        ]

    def _check_conditional(self, version):
        models = self._format_models()
        if not models:
            return []
        key = getattr(self, "HDUCLAS1", None)
        rules = (
            models.get("HEADER_CONDITIONAL_REQUIRED", {}).get(version, {}).get(key, [])
        )
        errors = []
        for field, value, req_if, req_if_not in rules:
            actual = getattr(self, field, None)
            required = req_if if actual == value else req_if_not
            missing = [k for k in required if getattr(self, k, None) is None]
            if missing:
                errors.append(f"{missing} required when {field}={actual!r}")
        return errors

    @model_validator(mode="after")
    def _enforce(self, info: ValidationInfo):
        ctx = info.context or {}
        strict = ctx.get("strict", False)
        version = (
            ctx.get("version")
            or getattr(self, "HDUVERS", None)
            or DEFAULT_DATA_FORMAT_VERSION
        )
        errors = []
        missing = self._check_required(version)
        if missing:
            errors.append(f"Missing mandatory keyword(s): {missing}")
        errors += self._check_conditional(version)
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


class CustomHDUHeader(HDUHeader, HDUResponseFields, GeneralFields):
    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_header(cls, header):
        return cls.model_construct(**dict(header))


# --------------- DISPATCH HELPERS ---------------
class UnknownHDUClass(IOError):
    """Raised when a file contains an unknown HDUCLASS."""


def _header_key(meta):
    """Helper function for the header class mapping."""
    return meta.get("HDUCLAS1") or meta.get("EXTNAME")


def _hdu_key(meta, fallback=None):
    """Helper function for the hdu dictionary keys."""
    hduclas2 = meta.get("HDUCLAS2")
    if hduclas2 is not None:
        return hduclas2.upper()
    return meta.get("EXTNAME") or fallback


def _hdu_class_key(meta):
    """Helper function for the gammapy class mapping"""
    # Imported lazily to avoid importing gammapy at module load.
    from gammapy.irf.io import _get_hdu_type_and_class

    hdu_class = meta.get("HDUCLAS4") or meta.get("HDUCLAS1") or meta.get("EXTNAME")
    if hdu_class not in PRODUCT_MODELS.keys():
        hdu_class = meta.get("HDUCLAS2", "unknown_clas")
        if (hdu_class in ["EFF_AREA", "RPSF", "EDISP", "BKG"]) and meta.get("HDUCLAS4"):
            _, hdu_class = _get_hdu_type_and_class(meta)  # CTA-1DC workaround
    return hdu_class.upper()


def _check_data_format(meta, format, strict):
    meta_format = meta.get("HDUCLASS")
    if meta_format != format and strict:
        raise ValueError(
            f"HDUReaderWriter expected a {format} HDU class, {meta_format} was provided."
        )
    return meta_format


def apply_meta_to_header(fits_header, header):
    meta = header.to_header()
    comments = meta.pop("comments", None)
    history = meta.pop("history", None)
    fits_header.update(meta)
    for line in ([comments] if isinstance(comments, str) else comments) or []:
        fits_header.add_comment(line)
    for line in ([history] if isinstance(history, str) else history) or []:
        fits_header.add_history(line)


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
        if not hasattr(header, "_check_required"):
            return None
        version = version or DEFAULT_DATA_FORMAT_VERSION
        errors = header._check_required(version) + header._check_conditional(version)
        if errors:
            msg = f"Header not compliant: {errors}"
            if strict:
                raise ValueError(msg)
            log.warning("%s (not enforced)", msg)
        return not errors

    @classmethod
    def validate_table(cls, table, format, version, strict):
        """Enforce the format's column schema on a table."""
        errors = []
        hdu_class_key = _hdu_class_key(table.meta)
        try:
            cls.table_validator(hdu=hdu_class_key, format=format, version=version).run(
                table
            )
        except (KeyError, TypeError, UnitTypeError) as e:
            errors.append(f"table: {e}")
        if errors:
            msg = f"{hdu_class_key} table not {format} compliant: " + ", ".join(errors)
            if strict:
                raise ValueError(msg)
            log.warning("%s (not enforced)", msg)
        return not errors

    @classmethod
    def format_validator(cls, header, table, format, version, strict):
        header_valid = cls.validate_header(header, version, strict)
        table_valid = cls.validate_table(table, format, version, strict)
        return header_valid, table_valid

    @classmethod
    def _from_fits_hdu(cls, fits_hdu, format, version, strict, verbose=True):
        """Build a reader/writer from an open FITS HDU (shared by read paths)."""
        if isinstance(fits_hdu, fits.PrimaryHDU):
            header = PrimaryHDUHeader.from_header(dict(fits_hdu.header))
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
            hkey, validate = "IMAGE", False
        else:
            table = Table.read(fits_hdu)
            meta, data = dict(table.meta), None
            hkey, validate = _header_key(meta), True

        hdu_class = _hdu_class_key(meta)
        detected = _check_data_format(meta, format, strict)
        if detected in (None, "CUSTOM"):
            meta_format = format
            log.warning(f"{hkey}: no detected format, using expected for validation.")
        else:
            meta_format = detected
        models = DATA_FORMATS_MODELS.get(meta_format)

        if models is None:  # declared a real but unregistered format
            return cls(
                table=table,
                data=data,
                header=CustomHDUHeader.from_header(meta),
                hdu=hdu_class,
                format=meta_format,
                version=None,
            )
        else:
            try:
                header = models["HEADER"][hkey].from_header(
                    meta, strict=strict, version=version
                )
            except (ValueError, KeyError, TypeError, UnitTypeError) as e:
                if strict:
                    raise
                if verbose:
                    log.warning(
                        "Header construction failed (%s), using CustomHDUHeader.", e
                    )
                header, version = CustomHDUHeader.from_header(meta), None
                header_valid, table_valid = False, False
            else:
                # construction succeeded -> keep this header; validation only reports
                if validate:
                    header_valid, table_valid = cls.format_validator(
                        header, table, meta_format, version, strict
                    )
                else:
                    header_valid, table_valid = None, None

        ReaderWriter = models.get("READER_WRITER", {}).get(hkey, cls)
        rw = ReaderWriter(
            table=table,
            data=data,
            header=header,
            hdu=hdu_class,
            format=meta_format,
            version=version,
        )
        rw._header_valid = header_valid
        rw._table_valid = table_valid
        return rw

        # try:
        #     header = models["HEADER"][hkey].from_header(meta, version, strict)
        #     if validate:
        #         cls.format_validator(header, table, meta_format, version, strict)
        # except (ValueError, KeyError, TypeError, UnitTypeError) as e:
        #     if strict:
        #         raise
        #     if verbose:
        #         log.warning("Header validation failed (%s), using CustomHDUHeader.", e)
        #     header, version = CustomHDUHeader.from_header(meta), None

        # return ReaderWriter(
        #     table=table, data=data, header=header,
        #     hdu=hdu_class, format=meta_format, version=version,
        # )

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
        from gammapy.io import gadf_metadata  # noqa: F401  (import side effect: registry injection)

        hdu = hdu or cls.DEFAULT_HDU
        if hdu is None:
            raise ValueError("HDUReaderWriter objects require an `hdu`.")
        with fits.open(make_path(filename), memmap=False) as hdulist:
            return cls._from_fits_hdu(hdulist[hdu], format, version, strict, verbose)

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
        self.format_validator(self.header, self.table, format, version, strict)
        table_hdu = fits.BinTableHDU(self.table, name=self.hdu)
        apply_meta_to_header(table_hdu.header, self.header)
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
    ):
        if self.table is not None:
            return self.to_table_hdu(format, version, strict)
        if self.data is not None:
            return self.to_image_hdu()
        raise ValueError(f"{self.hdu}: neither table nor data set, nothing to write.")

    def to_hdulist(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        return [fits.PrimaryHDU(), self._to_hdu(format, version, strict)]

    def write(
        self,
        filename,
        overwrite=False,
        checksum=False,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        hdulist = fits.HDUList(self.to_hdulist(format, version, strict))
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
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
        verbose=True,
    ):
        from gammapy.io import gadf_metadata  # noqa: F401  (import side effect: registry injection)

        filename = make_path(filename)
        if verbose:
            log.warning(
                "Reading %s \nExpected data format: %s v%s, strict=%s",
                filename,
                format,
                version,
                strict,
            )

        with fits.open(filename, memmap=False) as hdulist:
            hdu_dict = {}
            validation_info = []
            for hdu in hdulist:
                rw = HDUReaderWriter._from_fits_hdu(
                    hdu, format, version, strict, verbose
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
                header_row = (
                    f"{'HDU':<{kw}}  {'FMT':<{fw}} {'VER':<{vw}} {'HDR':<{hw}} {'TAB'}"
                )
                validation_lines = [
                    f"{k:<{kw}}  {f:<{fw}} {v:<{vw}} {h:<{hw}} {t}".rstrip()
                    for k, f, v, h, t in validation_info
                ]
                log.warning("%s\n%s", header_row, "\n".join(validation_lines))
        return cls(hdu_dict=hdu_dict)

    @property
    def format_list(self):
        return {key: rw.format for key, rw in self.hdu_dict.items()}

    def to_product_dict(self):
        product_dict = {}
        for hdu_name in self.hdu_dict:
            if hdu_name != "PRIMARY":
                product_dict[self.hdu_dict[hdu_name].hdu] = self.hdu_dict[
                    hdu_name
                ].to_product()
        return product_dict

    def to_hdulist(
        self,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_WRITE,
    ):
        hdulist = [fits.PrimaryHDU()]
        for hdu in self.hdu_dict:
            if hdu == "PRIMARY":
                continue
            hdulist.append(self.hdu_dict[hdu]._to_hdu(format, version, strict))
        return hdulist


class ProductReaderWriter:
    """IO for gammapy products (DL3/DL4/DL5)."""

    PRODUCT = None

    def __init__(
        self,
        filename=None,
        hdu=None,
        product=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
    ):
        from gammapy.io import gadf_metadata  # noqa: F401  (import side effect: registry injection)

        hdu = hdu or self.PRODUCT
        if product is None:
            rw = HDUListReaderWriter.read(filename, format, version, strict)
            product = rw.hdu_dict[hdu].to_product()
            print(product)
            if product is None:
                raise ValueError(f"HDU {hdu!r} did not resolve to a gammapy product.")
            self.product = product
            self.header = rw.hdu_dict[hdu].header
            format = rw.hdu_dict[hdu].format
        else:
            self.product = product
            meta = product.meta
            self.header = DATA_FORMATS_MODELS[format]["HEADER"][
                _header_key(meta)
            ].from_header(meta, version, strict)
        self.format = format
        self.version = version

    @classmethod
    def read(
        cls,
        filename,
        hdu=None,
        format=DEFAULT_DATA_FORMAT,
        version=DEFAULT_DATA_FORMAT_VERSION,
        strict=DEFAULT_STRICT_READ,
    ):
        hdu = hdu or cls.PRODUCT
        return cls(filename, hdu, None, format, version, strict).product

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
            HDUReaderWriter.validate_header(self.header, strict)
        for hdu in hdulist:
            if isinstance(hdu, fits.PrimaryHDU) or hdu.is_image:
                continue
            HDUReaderWriter.validate_table(Table.read(hdu), format, version, strict)

        hdulist.writeto(
            str(make_path(filename)), overwrite=overwrite, checksum=checksum
        )
