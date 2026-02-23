"""Sensor platform for Sun Shade Integration."""

from homeassistant.components.sensor import (
    SensorEntity,
    SensorStateClass,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import DEGREE, PERCENTAGE
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
)

from .const import DOMAIN, CONF_WINDOWS


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
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_intensity"
        self._attr_name = f"{self._window_id} sun intensity"

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
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_angle"
        self._attr_name = f"{self._window_id} sun angle"

    @property
    def native_value(self) -> float | None:
        """Return sun angle to window normal in degrees."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        return detail["sun_angle_to_normal_deg"]
