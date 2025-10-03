"""Ephys dataset structure and runtime configuration.

This module defines `EphysData`, a dataclass used as an in-memory
container for electrophysiology experiment data and related runtime
parameters. It is intended as the single place to hold data that
notebooks and scripts commonly share during analysis.

Main responsibilities:
- Store identifiers and paths (KSfolder, ephysFolder, rat, date).
- Hold spike and waveform data (per-spike arrays in `allSpikeSI`,
    `allSpike_amp`, `allSpikeTimes`, `waveform`).
- Track per-cell metadata (`cellNumbers`, `goodCells`, `ks_ids`).
- Represent flagged/bad periods in `buggy_periods` (dict of arrays).
- Provide helpers: `reset()` to restore defaults, `as_dict()` to
    create a JSON-serializable snapshot, and `summary()` for quick
    inspection in notebooks.

Design notes:
- Numeric sequences use numpy arrays for performance and interoperability.
- `allSpikeSI` is a list of numpy arrays to allow variable-length per-spike
    feature vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List
import numpy as np


def _empty_float_array() -> np.ndarray:
    return np.array([], dtype=float)


def _empty_int_array() -> np.ndarray:
    return np.array([], dtype=int)


def _empty_obj_array() -> np.ndarray:
    return np.array([], dtype=object)


@dataclass
class EphysData:
    # Paths / identifiers
    KSfolder: Optional[str] = None
    ephysFolder: Optional[str] = None
    rat: Optional[str] = None
    date: Optional[str] = None

    # Spike / waveform related
    # buggy_periods: mapping from label -> Nx2 numpy array of start/end
    buggy_periods: Dict[str, np.ndarray] = field(default_factory=dict)
    allSpikeSI: List[np.ndarray] = field(default_factory=list)
    allSpike_amp: np.ndarray = field(default_factory=_empty_float_array)
    waveform: np.ndarray = field(default_factory=_empty_float_array)
    allSpikeTimes: List[np.ndarray] = field(default_factory=list)
    binnedSpikesOriginal: np.ndarray = field(default_factory=_empty_float_array)

    # Cell / channel metadata

    cellNumbers: np.ndarray = field(default_factory=_empty_int_array)
    goodCells: np.ndarray = field(default_factory=_empty_int_array)
    goodCells2: np.ndarray = field(default_factory=_empty_int_array)
    goodCellsIndices: np.ndarray = field(default_factory=_empty_int_array)
    ks_ids: np.ndarray = field(default_factory=_empty_int_array)
    ML: np.ndarray = field(default_factory=_empty_int_array)
    DV: np.ndarray = field(default_factory=_empty_int_array)
    Amplitude: np.ndarray = field(default_factory=_empty_int_array)
    channel: np.ndarray = field(default_factory=_empty_int_array)
    fr: np.ndarray = field(default_factory=_empty_int_array)

    # Misc numeric / params
    toLoadLFP: bool = False
    timeAroundPokeWindows: Optional[float] = None
    spikeBinSizeWindows: Optional[float] = None
    binsToAverageWindows: int = 1
    lastSpikeTime: Optional[float] = None

    # Windows
    windows: List[np.ndarray] = field(default_factory=list)
    windowsOri: List[np.ndarray] = field(default_factory=list)
    windowsRec: List[np.ndarray] = field(default_factory=list)


    def reset(self) -> None:
        """Reset this config instance to its dataclass defaults."""
        defaults = type(self)()
        # replace in-place so references to this instance remain valid
        self.__dict__.clear()
        self.__dict__.update(defaults.__dict__)

    def append_allSpikeSI(self, arr: np.ndarray) -> None:
        """Append a numpy array to the per-spike SI list.

        Ensures the stored object is a numpy array.
        """
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr)
        self.allSpikeSI.append(arr)

    def validate(self) -> None:
        """Run lightweight validation of common fields.

        Raises ValueError on obvious inconsistencies.
        """
        # cellNumbers and ks_ids lengths should match when both present
        if isinstance(self.cellNumbers, np.ndarray) and isinstance(self.ks_ids, np.ndarray):
            if self.cellNumbers.size != self.ks_ids.size:
                raise ValueError("cellNumbers and ks_ids must have the same length")

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict representation.

        Numpy arrays are converted to lists.
        """
        out: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            # numpy arrays -> lists
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
                continue

            # lists of numpy arrays -> lists of lists
            if isinstance(v, list) and all(isinstance(x, np.ndarray) for x in v):
                out[k] = [x.tolist() for x in v]
                continue

            # dicts with numpy array values -> dict of lists
            if isinstance(v, dict) and all(isinstance(x, np.ndarray) for x in v.values()):
                out[k] = {kk: vv.tolist() for kk, vv in v.items()}
                continue

            # other python objects
            out[k] = v
        return out

    def summary(self) -> Dict[str, Any]:
        """Small summary useful in notebooks/logs."""
        return {
            "KSfolder": self.KSfolder,
            "ephysFolder": self.ephysFolder,
            "rat": self.rat,
            "date": self.date,
            "n_cells": int(self.cellNumbers.size) if isinstance(self.cellNumbers, np.ndarray) else 0,
            "n_good_cells": int(self.goodCells.size) if isinstance(self.goodCells, np.ndarray) else 0,
            "lastSpikeTime": self.lastSpikeTime,
        }


# Module-level singleton for convenience in notebooks and scripts.
e = EphysData()

__all__ = ["EphysData", "e"]
