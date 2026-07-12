"""Industrial inspection device catalog and task-context data models."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union


Point2D = Tuple[float, float]


def _point2(value: Sequence[float], *, field: str) -> Point2D:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain exactly two coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{field} coordinates must be finite")
    return point


def _finite(value: Any, *, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


@dataclass(frozen=True)
class GroundStationSpec:
    position: Point2D
    los_anchor: Point2D

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "GroundStationSpec":
        position = _point2(data.get("position"), field="ground_station.position")
        anchor = _point2(data.get("los_anchor", position), field="ground_station.los_anchor")
        return cls(position=position, los_anchor=anchor)


@dataclass(frozen=True)
class DeviceObservationSpec:
    min_distance: float
    max_distance: float
    preferred_bearing_rad: float
    bearing_tolerance_rad: float
    fov_angle_rad: float
    require_los: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, device_id: str) -> "DeviceObservationSpec":
        prefix = f"devices[{device_id}].observation"
        min_distance = _finite(data.get("min_distance"), field=f"{prefix}.min_distance")
        max_distance = _finite(data.get("max_distance"), field=f"{prefix}.max_distance")
        preferred = _finite(
            data.get("preferred_bearing_rad"),
            field=f"{prefix}.preferred_bearing_rad",
        )
        tolerance = _finite(
            data.get("bearing_tolerance_rad"),
            field=f"{prefix}.bearing_tolerance_rad",
        )
        fov = _finite(data.get("fov_angle_rad"), field=f"{prefix}.fov_angle_rad")
        if min_distance < 0.0:
            raise ValueError(f"{prefix}.min_distance must be nonnegative")
        if max_distance <= min_distance:
            raise ValueError(f"{prefix}.max_distance must exceed min_distance")
        if not -math.pi <= preferred <= math.pi:
            raise ValueError(f"{prefix}.preferred_bearing_rad must be in [-pi, pi]")
        if not 0.0 < tolerance <= math.pi:
            raise ValueError(f"{prefix}.bearing_tolerance_rad must be in (0, pi]")
        if not 0.0 < fov <= 2.0 * math.pi:
            raise ValueError(f"{prefix}.fov_angle_rad must be in (0, 2*pi]")
        require_los = data.get("require_los")
        if not isinstance(require_los, bool):
            raise ValueError(f"{prefix}.require_los must be boolean")
        return cls(
            min_distance=min_distance,
            max_distance=max_distance,
            preferred_bearing_rad=preferred,
            bearing_tolerance_rad=tolerance,
            fov_angle_rad=fov,
            require_los=require_los,
        )


@dataclass(frozen=True)
class DeviceTaskSpec:
    device_id: str
    position: Point2D
    observation_anchor: Point2D
    observation: DeviceObservationSpec

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeviceTaskSpec":
        device_id = str(data.get("id", "")).strip()
        if not device_id:
            raise ValueError("device id must be a non-empty string")
        observation_data = data.get("observation")
        if not isinstance(observation_data, Mapping):
            raise ValueError(f"devices[{device_id}].observation must be an object")
        return cls(
            device_id=device_id,
            position=_point2(data.get("position"), field=f"devices[{device_id}].position"),
            observation_anchor=_point2(
                data.get("observation_anchor"),
                field=f"devices[{device_id}].observation_anchor",
            ),
            observation=DeviceObservationSpec.from_dict(observation_data, device_id=device_id),
        )


@dataclass(frozen=True)
class IndustrialInspectionCatalog:
    ground_station: GroundStationSpec
    devices: Tuple[DeviceTaskSpec, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IndustrialInspectionCatalog":
        station_data = data.get("ground_station")
        if not isinstance(station_data, Mapping):
            raise ValueError("ground_station must be an object")
        raw_devices = data.get("devices")
        if not isinstance(raw_devices, list) or not raw_devices:
            raise ValueError("devices must be a non-empty list")
        devices = tuple(DeviceTaskSpec.from_dict(item) for item in raw_devices)
        ids = [device.device_id for device in devices]
        duplicates = sorted({device_id for device_id in ids if ids.count(device_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate device id(s): {', '.join(duplicates)}")
        return cls(
            ground_station=GroundStationSpec.from_dict(station_data),
            devices=devices,
        )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "IndustrialInspectionCatalog":
        catalog_path = Path(path)
        with catalog_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, Mapping):
            raise ValueError("device catalog root must be a JSON object")
        return cls.from_dict(data)

    def get_device(self, device_id: str) -> DeviceTaskSpec:
        for device in self.devices:
            if device.device_id == str(device_id):
                return device
        known = ", ".join(device.device_id for device in self.devices)
        raise KeyError(f"unknown device_id={device_id!r}; known devices: {known}")


CatalogInput = Union[str, Path, Mapping[str, Any], IndustrialInspectionCatalog]


def load_device_catalog(value: CatalogInput) -> IndustrialInspectionCatalog:
    if isinstance(value, IndustrialInspectionCatalog):
        return value
    if isinstance(value, Mapping):
        return IndustrialInspectionCatalog.from_dict(value)
    return IndustrialInspectionCatalog.from_json(value)


class TaskContextInfeasibleError(RuntimeError):
    """Raised when a requested device has no state in its task-terminal set."""

