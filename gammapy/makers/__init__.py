from gammapy.utils.registry import Registry

from .core import Maker
from .background import (
    AdaptiveRingBackgroundMaker,
    FoVBackgroundMaker,
    PhaseBackgroundMaker,
    ReflectedRegionsBackgroundMaker,
    ReflectedRegionsFinder,
    RegionsFinder,
    RingBackgroundMaker,
    WobbleRegionsFinder,
)

from .map import MapDatasetMaker, HessMapDatasetMaker
from .reduce import DatasetsMaker
from .safe import SafeMaskMaker
from .spectrum import SpectrumDatasetMaker

MAKER_REGISTRY = Registry(
    [
        ReflectedRegionsBackgroundMaker,
        AdaptiveRingBackgroundMaker,
        FoVBackgroundMaker,
        PhaseBackgroundMaker,
        RingBackgroundMaker,
        SpectrumDatasetMaker,
        MapDatasetMaker,
        SafeMaskMaker,
        DatasetsMaker,
        HessMapDatasetMaker,
    ]
)
"""Registry of maker classes in Gammapy."""

__all__ = [
    "AdaptiveRingBackgroundMaker",
    "DatasetsMaker",
    "FoVBackgroundMaker",
    "HessMapDatasetMaker",
    "Maker",
    "MAKER_REGISTRY",
    "MapDatasetMaker",
    "PhaseBackgroundMaker",
    "ReflectedRegionsBackgroundMaker",
    "ReflectedRegionsFinder",
    "RegionsFinder",
    "RingBackgroundMaker",
    "SafeMaskMaker",
    "SpectrumDatasetMaker",
    "WobbleRegionsFinder",
]
