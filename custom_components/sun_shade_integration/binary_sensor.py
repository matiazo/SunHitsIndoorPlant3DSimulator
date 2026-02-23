"""Binary sensor platform for Sun Shade Integration."""

from homeassistant.components.binary_sensor import (
    BinarySensorDeviceClass,
    BinarySensorEntity,
)
from homeassistant.config_entries import ConfigEntry
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
    """Set up binary sensors from a config entry."""
    coordinator: DataUpdateCoordinator = hass.data[DOMAIN][entry.entry_id]
    windows = entry.data.get(CONF_WINDOWS, [])

    entities = [
        WindowHasSunBinarySensor(coordinator, entry, window)
        for window in windows
    ]
    async_add_entities(entities)


class WindowHasSunBinarySensor(CoordinatorEntity, BinarySensorEntity):
    """Binary sensor indicating whether a window is receiving direct sun."""

    _attr_device_class = BinarySensorDeviceClass.LIGHT

    def __init__(
        self,
        coordinator: DataUpdateCoordinator,
        entry: ConfigEntry,
        window: dict,
    ) -> None:
        """Initialize the binary sensor."""
        super().__init__(coordinator)
        self._window_id = window["id"]
        self._attr_unique_id = f"{entry.entry_id}_{self._window_id}_has_sun"
        self._attr_name = f"{self._window_id} has sun"

    @property
    def is_on(self) -> bool | None:
        """Return True if window has sun."""
        if self.coordinator.data is None:
            return None
        detail = self.coordinator.data.get(self._window_id)
        if detail is None:
            return None
        return detail["is_in_sun"]
