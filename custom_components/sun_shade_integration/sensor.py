"""Sensor platform for Sun Shade Integration."""

from datetime import datetime

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE, PERCENTAGE, UnitOfTime
from homeassistant.core import HomeAssistant
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)
from homeassistant.util import dt as dt_util

from .const import DOMAIN, CONF_WINDOWS


def _time_str_to_datetime(time_str: str) -> datetime:
    """Convert HH:MM string to today's datetime in HA's local timezone.

    Constructs a timezone-aware datetime using HA's configured timezone,
    which correctly handles DST transitions.
    """
    now = dt_util.now()
    hour, minute = map(int, time_str.split(":"))
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensors from a config entry."""
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    windows = entry.data.get(CONF_WINDOWS, [])

    entities: list[SensorEntity] = []
    for window in windows:
        entities.append(WindowSunIntensitySensor(coordinator, entry, window))
        entities.append(WindowSunAngleSensor(coordinator, entry, window))
        entities.append(WindowFirstLightSensor(coordinator, entry, window))
        entities.append(WindowLastLightSensor(coordinator, entry, window))

    # Plant-level daily forecast sensors
    entities.append(PlantSunStartSensor(coordinator, entry))
    entities.append(PlantSunEndSensor(coordinator, entry))
    entities.append(PlantSunDurationSensor(coordinator, entry))

    async_add_entities(entities)


class WindowSunIntensitySensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting sun intensity factor for a window (0.0-1.0)."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = PERCENTAGE
    _attr_suggested_display_precision = 1

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
        window: dict,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_intensity"
        self._attr_name = f"{self._window_id} sun intensity"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group under window device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry_id}_{self._window_id}")},
            name=f"Window {self._window_id}",
            manufacturer="Sun Shade Integration",
            model="Window Sun Sensor",
        )

    @property
    def native_value(self) -> float | None:
        """Return sun intensity as percentage (0-100)."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        return round(detail["intensity_factor"] * 100, 1)


class WindowSunAngleSensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting sun angle to window normal (0-90 degrees)."""

    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_native_unit_of_measurement = DEGREE

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
        window: dict,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_angle"
        self._attr_name = f"{self._window_id} sun angle"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group under window device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry_id}_{self._window_id}")},
            name=f"Window {self._window_id}",
            manufacturer="Sun Shade Integration",
            model="Window Sun Sensor",
        )

    @property
    def native_value(self) -> float | None:
        """Return sun angle to window normal in degrees."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        return detail["sun_angle_to_normal_deg"]


class WindowFirstLightSensor(CoordinatorEntity, SensorEntity):
    """Forecast: first time sun hits plant through this window today."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
        window: dict,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_first_light"
        self._attr_name = f"{self._window_id} first light"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group under window device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry_id}_{self._window_id}")},
            name=f"Window {self._window_id}",
            manufacturer="Sun Shade Integration",
            model="Window Sun Sensor",
        )

    @property
    def native_value(self) -> datetime | None:
        """Return first time sun hits plant through this window today."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        time_str = detail.get("first_light")
        if time_str is None:
            return None
        return _time_str_to_datetime(time_str)


class WindowLastLightSensor(CoordinatorEntity, SensorEntity):
    """Forecast: last time sun hits plant through this window today."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
        window: dict,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_last_light"
        self._attr_name = f"{self._window_id} last light"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info to group under window device."""
        return DeviceInfo(
            identifiers={(DOMAIN, f"{self._entry_id}_{self._window_id}")},
            name=f"Window {self._window_id}",
            manufacturer="Sun Shade Integration",
            model="Window Sun Sensor",
        )

    @property
    def native_value(self) -> datetime | None:
        """Return last time sun hits plant through this window today."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        time_str = detail.get("last_light")
        if time_str is None:
            return None
        return _time_str_to_datetime(time_str)


class PlantSunStartSensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting when sun first hits the plant today (like sunrise)."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_plant_sun_start"
        self._attr_name = "Plant sun start"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the plant device."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="Sun Shade Plant",
            manufacturer="Sun Shade Integration",
            model="Plant Sun Sensor",
        )

    @property
    def native_value(self) -> datetime | None:
        """Return first time sun hits the plant today as a datetime."""
        forecast = self._get_forecast()
        if forecast is None or forecast.get("sun_start") is None:
            return None
        return _time_str_to_datetime(forecast["sun_start"])

    def _get_forecast(self) -> dict | None:
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get("_plant_forecast")


class PlantSunEndSensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting when sun last hits the plant today (like sunset)."""

    _attr_device_class = SensorDeviceClass.TIMESTAMP

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_plant_sun_end"
        self._attr_name = "Plant sun end"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the plant device."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="Sun Shade Plant",
            manufacturer="Sun Shade Integration",
            model="Plant Sun Sensor",
        )

    @property
    def native_value(self) -> datetime | None:
        """Return last time sun hits the plant today as a datetime."""
        forecast = self._get_forecast()
        if forecast is None or forecast.get("sun_end") is None:
            return None
        return _time_str_to_datetime(forecast["sun_end"])

    def _get_forecast(self) -> dict | None:
        if self.coordinator.data is None:
            return None
        return self.coordinator.data.get("_plant_forecast")


class PlantSunDurationSensor(CoordinatorEntity, SensorEntity):
    """Sensor reporting total sun exposure duration on the plant today."""

    _attr_native_unit_of_measurement = UnitOfTime.MINUTES
    _attr_device_class = SensorDeviceClass.DURATION
    _attr_state_class = SensorStateClass.TOTAL

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
    ) -> None:
        """Initialize the sensor."""
        super().__init__(coordinator)
        self._entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_plant_sun_duration"
        self._attr_name = "Plant sun duration"

    @property
    def device_info(self) -> DeviceInfo:
        """Return device info for the plant device."""
        return DeviceInfo(
            identifiers={(DOMAIN, self._entry_id)},
            name="Sun Shade Plant",
            manufacturer="Sun Shade Integration",
            model="Plant Sun Sensor",
        )

    @property
    def native_value(self) -> int | None:
        """Return total sun exposure in minutes."""
        if self.coordinator.data is None:
            return None
        forecast = self.coordinator.data.get("_plant_forecast")
        if forecast is None:
            return None
        return forecast.get("sun_duration_min", 0)
