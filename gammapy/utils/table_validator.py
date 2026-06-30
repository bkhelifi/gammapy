# Licensed under a 3-clause BSD style license - see LICENSE.rst
from collections.abc import MutableMapping
from typing import Optional, Tuple, Union

import numpy as np
import astropy.units as u
import yaml
from astropy.table import Column, Table
from astropy.units import Unit, UnitTypeError
from pydantic import BaseModel, ConfigDict, field_validator


class ColumnDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dtype: Optional[str] = None
    unit: Optional[Union[Unit, str]] = u.one
    shape: Tuple = ()
    ndim: Optional[int] = None
    description: Optional[str] = None
    required: bool = False

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, v):
        if v is None:
            return u.one
        return Unit(v)

    def validate_column(self, column):
        self.check_column_ndim(self.ndim, column)
        self.check_column_unit(self.unit, column)
        return column

    @staticmethod
    def check_column_ndim(ndim, column):
        if ndim is None:
            return True
        found = column.ndim - 1  # drop the row axis
        if found == ndim:
            return True
        raise TypeError(f"Column ndim incorrect. Expected {ndim}, got {found}.")

    @staticmethod
    def check_column_type(dtype, column):
        if dtype is None or np.dtype(column.dtype).kind == np.dtype(dtype).kind:
            return True
        raise TypeError(
            f"Column dtype incorrect. Expected kind of {dtype}, got {column.dtype}."
        )

    @staticmethod
    def check_column_shape(shape, column):
        # column.shape is (n_rows, *element_shape); compare the per-row part.
        if tuple(column.shape[1:]) == tuple(shape):
            return True
        raise TypeError(
            f"Column shape incorrect. Expected {tuple(shape)}, got {tuple(column.shape[1:])}."
        )

    @staticmethod
    def check_column_unit(unit, column):
        expected = u.one if unit is None else Unit(unit)
        found = u.one if column.unit is None else column.unit
        if found.is_equivalent(expected):
            return True
        raise UnitTypeError(
            f"Column unit incorrect. Expected {expected}, got {column.unit}."
        )

    def to_column(self, name):
        """Build an empty `~astropy.table.Column` from this definition."""
        return Column(
            name=name,
            dtype=self.dtype,
            unit=self.unit,
            ndim=self.ndim,
            shape=self.shape,
            description=self.description,
        )


class TableValidator(MutableMapping):
    """A mapping of column name -> `ColumnDefinition`, with table validation."""

    def __init__(self, **kwargs):
        self._data = {}
        for key, value in kwargs.items():
            self[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __delitem__(self, key):
        del self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, ColumnDefinition):
            self._data[key] = value
        else:
            raise TypeError(f"Invalid type: {type(value)!r}")

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def to_yaml(self):
        res = {}
        for key, cdef in self._data.items():
            entry = {
                "dtype": cdef.dtype,
                "unit": str(cdef.unit),
                "required": cdef.required,
            }
            if cdef.shape != ():
                entry["shape"] = list(cdef.shape)
            if cdef.description is not None:
                entry["description"] = cdef.description
            res[key] = entry
        return yaml.dump(
            res, sort_keys=False, indent=4, width=80, default_flow_style=False
        )

    @classmethod
    def from_dict(cls, coldefs):
        res = cls()
        for key, item in coldefs.items():
            res[key] = ColumnDefinition(**(item or {}))
        return res

    @classmethod
    def from_yaml(cls, yaml_str):
        return cls.from_dict(yaml.safe_load(yaml_str))

    @property
    def required_columns(self):
        return [key for key in self._data if self[key].required]

    @property
    def optional_columns(self):
        return [key for key in self._data if not self[key].required]

    def run(self, table):
        """Validate a table against the definition; returns the table."""
        for key in self.required_columns:
            if key not in table.colnames:
                raise KeyError(f"Missing required column: {key!r}")
            self[key].validate_column(table[key])
        for key in self.optional_columns:
            if key in table.colnames:
                self[key].validate_column(table[key])
        return table

    def to_table(self, include_optional=None):
        """Build an empty table from the definition."""
        data = {}
        if include_optional is None:
            include_optional = []
        elif include_optional == "all":
            include_optional = self.optional_columns

        for key in self._data:
            if key in include_optional or self[key].required:
                data[key] = self[key].to_column(name=key)
        return Table(data)
