from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from pyace.const import EWEIGHTS_COL, ENERGY_CORRECTED_COL, FIT_SOURCE_ENERGY_OFFSETS_KW, NUMBER_OF_ATOMS


@dataclass
class SourceEnergyOffsetManager:
    column: Optional[str] = None
    reference_value: Any = None
    enabled: bool = False
    last_offsets: Dict[Any, float] = field(default_factory=dict)
    last_reference_value: Any = None

    @classmethod
    def from_fit_config(cls, fit_config: Optional[Dict]) -> "SourceEnergyOffsetManager":
        if not fit_config:
            return cls()

        spec = fit_config.get(FIT_SOURCE_ENERGY_OFFSETS_KW)
        if not spec:
            return cls()

        if isinstance(spec, str):
            return cls(column=spec, enabled=True)

        if isinstance(spec, dict):
            column = spec.get("column") or spec.get("group_by")
            if not column:
                raise ValueError(
                    f"`fit::{FIT_SOURCE_ENERGY_OFFSETS_KW}` requires a `column` entry. "
                    f"Got: {spec}"
                )
            reference_value = spec.get("reference")
            if reference_value is None:
                reference_value = spec.get("reference_value")
            return cls(column=column, reference_value=reference_value, enabled=True)

        raise ValueError(
            f"`fit::{FIT_SOURCE_ENERGY_OFFSETS_KW}` should be either a string column name or a mapping. "
            f"Got {type(spec)}"
        )

    def has_labels(self, dataframe: Optional[pd.DataFrame]) -> bool:
        return bool(self.enabled and dataframe is not None and self.column in dataframe.columns)

    def validate_dataframe(self, dataframe: Optional[pd.DataFrame], dataset_name: str = "dataset") -> None:
        if not self.enabled:
            return
        if dataframe is None:
            raise ValueError(
                f"`fit::{FIT_SOURCE_ENERGY_OFFSETS_KW}` is enabled, but {dataset_name} is None"
            )
        if self.column not in dataframe.columns:
            raise ValueError(
                f"`fit::{FIT_SOURCE_ENERGY_OFFSETS_KW}` expects column `{self.column}` "
                f"in the {dataset_name}"
            )
        if dataframe[self.column].isna().any():
            raise ValueError(
                f"`fit::{FIT_SOURCE_ENERGY_OFFSETS_KW}` does not support missing values in "
                f"column `{self.column}`"
            )

    def _get_reference_value(self, dataframe: pd.DataFrame):
        labels = pd.unique(dataframe[self.column])
        if len(labels) == 0:
            raise ValueError(f"No labels found in column `{self.column}`")

        if self.reference_value is None:
            reference_value = labels[0]
        else:
            reference_value = self.reference_value
            if reference_value not in set(labels.tolist()):
                raise ValueError(
                    f"Reference source label `{reference_value}` is not present in column `{self.column}`"
                )
        self.last_reference_value = reference_value
        return reference_value

    @staticmethod
    def _as_flat_float_array(values) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        return array.reshape(-1)

    def get_structure_weights(self, dataframe: pd.DataFrame) -> np.ndarray:
        if EWEIGHTS_COL in dataframe.columns:
            weights = np.vstack(dataframe[EWEIGHTS_COL].to_numpy()).reshape(-1)
        else:
            weights = np.ones(len(dataframe), dtype=float)
        return self._as_flat_float_array(weights)

    def fit_offsets(self, dataframe: pd.DataFrame, energy_pred) -> Dict[Any, float]:
        self.validate_dataframe(dataframe, dataset_name="fitting dataframe")

        nat = self._as_flat_float_array(dataframe[NUMBER_OF_ATOMS].to_numpy())
        labels = dataframe[self.column].to_numpy()
        energy_true = self._as_flat_float_array(dataframe[ENERGY_CORRECTED_COL].to_numpy())
        energy_pred = self._as_flat_float_array(energy_pred)
        if len(energy_pred) != len(dataframe):
            raise ValueError(
                f"Predicted energies size mismatch for source offsets: {len(energy_pred)} != {len(dataframe)}"
            )

        reference_value = self._get_reference_value(dataframe)
        weights = self.get_structure_weights(dataframe)
        residual_per_atom = (energy_true - energy_pred) / nat

        offsets = {}
        for label in pd.unique(labels):
            if label == reference_value:
                offsets[label] = 0.0
                continue

            mask = labels == label
            label_weights = weights[mask]
            if np.sum(label_weights) > 0:
                offsets[label] = float(np.average(residual_per_atom[mask], weights=label_weights))
            else:
                offsets[label] = float(np.mean(residual_per_atom[mask]))

        self.last_offsets = offsets
        return offsets

    def get_offsets_per_atom(self, dataframe: pd.DataFrame, offsets: Optional[Dict[Any, float]] = None) -> np.ndarray:
        if offsets is None:
            offsets = self.last_offsets
        if not offsets:
            return np.zeros(len(dataframe), dtype=float)
        self.validate_dataframe(dataframe, dataset_name="prediction dataframe")
        labels = dataframe[self.column].to_numpy()
        try:
            return np.array([offsets[label] for label in labels], dtype=float)
        except KeyError as exc:
            missing_label = exc.args[0]
            raise ValueError(
                f"Label `{missing_label}` from column `{self.column}` was not seen in the fitted source offsets"
            ) from exc

    def get_total_offsets(self, dataframe: pd.DataFrame, offsets: Optional[Dict[Any, float]] = None) -> np.ndarray:
        nat = self._as_flat_float_array(dataframe[NUMBER_OF_ATOMS].to_numpy())
        return self.get_offsets_per_atom(dataframe, offsets=offsets) * nat

    def get_adjusted_target_energies(
        self, dataframe: pd.DataFrame, offsets: Optional[Dict[Any, float]] = None
    ) -> np.ndarray:
        energy_true = self._as_flat_float_array(dataframe[ENERGY_CORRECTED_COL].to_numpy())
        return energy_true - self.get_total_offsets(dataframe, offsets=offsets)

    def get_adjusted_target_energies_per_atom(
        self, dataframe: pd.DataFrame, offsets: Optional[Dict[Any, float]] = None
    ) -> np.ndarray:
        nat = self._as_flat_float_array(dataframe[NUMBER_OF_ATOMS].to_numpy())
        return self.get_adjusted_target_energies(dataframe, offsets=offsets) / nat

    def apply_to_predictions(
        self, energy_pred, dataframe: pd.DataFrame, offsets: Optional[Dict[Any, float]] = None
    ) -> np.ndarray:
        energy_pred = self._as_flat_float_array(energy_pred)
        return energy_pred + self.get_total_offsets(dataframe, offsets=offsets)

    def serialize_offsets(self, offsets: Optional[Dict[Any, float]] = None) -> Dict[str, float]:
        if offsets is None:
            offsets = self.last_offsets
        return {str(label): float(value) for label, value in offsets.items()}
