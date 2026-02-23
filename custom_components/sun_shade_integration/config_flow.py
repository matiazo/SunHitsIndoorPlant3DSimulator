"""Config flow for Sun Shade Integration."""

import logging
from typing import Any

import voluptuous as vol

from homeassistant import config_entries
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    BooleanSelector,
    EntitySelector,
    EntitySelectorConfig,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
)

from .const import (
    DOMAIN,
    CONF_UPDATE_INTERVAL,
    CONF_WALLS,
    CONF_WINDOWS,
    CONF_PLANT,
    CONF_WALL_ID,
    CONF_OUTWARD_NORMAL,
    CONF_WALL_THICKNESS,
    CONF_WALL_AXIS,
    CONF_WINDOW_COUNT,
    CONF_DEFAULT_WINDOW_WIDTH,
    CONF_DEFAULT_WINDOW_HEIGHT,
    CONF_DEFAULT_Z_BOTTOM,
    CONF_DEFAULT_Z_TOP,
    CONF_ADD_ANOTHER_WALL,
    CONF_WINDOW_ID,
    CONF_POSITION_ALONG_WALL,
    CONF_WINDOW_WIDTH,
    CONF_WINDOW_HEIGHT,
    CONF_Z_BOTTOM,
    CONF_Z_TOP,
    CONF_SHADE_ENTITY_ID,
    CONF_PLANT_DIST_WALL1,
    CONF_PLANT_DIST_WALL2,
    CONF_PLANT_RADIUS,
    CONF_PLANT_Z_MIN,
    CONF_PLANT_Z_MAX,
    DEFAULT_UPDATE_INTERVAL,
)

_LOGGER = logging.getLogger(__name__)


def _user_schema(defaults: dict | None = None) -> vol.Schema:
    """Build the general settings schema."""
    d = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_UPDATE_INTERVAL,
                default=d.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=30, max=3600, step=1, mode=NumberSelectorMode.BOX
                )
            ),
        }
    )


def _wall_schema(wall_number: int, defaults: dict | None = None) -> vol.Schema:
    """Build the wall definition schema."""
    d = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_WALL_ID,
                default=d.get(CONF_WALL_ID, f"wall_{wall_number}"),
            ): TextSelector(TextSelectorConfig()),
            vol.Required(
                CONF_OUTWARD_NORMAL,
                default=d.get(CONF_OUTWARD_NORMAL, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=360, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WALL_THICKNESS,
                default=d.get(CONF_WALL_THICKNESS, 0.25),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=1.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WALL_AXIS,
                default=d.get(CONF_WALL_AXIS, "x"),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["x", "y"],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Required(
                CONF_WINDOW_COUNT,
                default=d.get(CONF_WINDOW_COUNT, 1),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=1, max=10, step=1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_DEFAULT_WINDOW_WIDTH,
                default=d.get(CONF_DEFAULT_WINDOW_WIDTH, 0.89),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.01, max=10.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_DEFAULT_WINDOW_HEIGHT,
                default=d.get(CONF_DEFAULT_WINDOW_HEIGHT, 1.50),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.01, max=10.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_DEFAULT_Z_BOTTOM,
                default=d.get(CONF_DEFAULT_Z_BOTTOM, 4.2),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_DEFAULT_Z_TOP,
                default=d.get(CONF_DEFAULT_Z_TOP, 5.7),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_ADD_ANOTHER_WALL, default=False
            ): BooleanSelector(),
        }
    )


def _edit_wall_schema(wall_number: int, defaults: dict | None = None) -> vol.Schema:
    """Build the wall edit schema (no add_another_wall or window_count)."""
    d = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_WALL_ID,
                default=d.get(CONF_WALL_ID, f"wall_{wall_number}"),
            ): TextSelector(TextSelectorConfig()),
            vol.Required(
                CONF_OUTWARD_NORMAL,
                default=d.get(CONF_OUTWARD_NORMAL, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=360, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WALL_THICKNESS,
                default=d.get(CONF_WALL_THICKNESS, 0.25),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=1.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WALL_AXIS,
                default=d.get(CONF_WALL_AXIS, "x"),
            ): SelectSelector(
                SelectSelectorConfig(
                    options=["x", "y"],
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
    )


def _window_schema(defaults: dict | None = None) -> vol.Schema:
    """Build the per-window schema."""
    d = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_WINDOW_ID,
                default=d.get(CONF_WINDOW_ID, ""),
            ): TextSelector(TextSelectorConfig()),
            vol.Required(
                CONF_POSITION_ALONG_WALL,
                default=d.get(CONF_POSITION_ALONG_WALL, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=100.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WINDOW_WIDTH,
                default=d.get(CONF_WINDOW_WIDTH, 0.89),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.01, max=10.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_WINDOW_HEIGHT,
                default=d.get(CONF_WINDOW_HEIGHT, 1.50),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.01, max=10.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_Z_BOTTOM,
                default=d.get(CONF_Z_BOTTOM, 4.2),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_Z_TOP,
                default=d.get(CONF_Z_TOP, 5.7),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Optional(
                CONF_SHADE_ENTITY_ID,
                default=d.get(CONF_SHADE_ENTITY_ID, ""),
            ): EntitySelector(EntitySelectorConfig(domain="cover")),
        }
    )


def _plant_schema(defaults: dict | None = None) -> vol.Schema:
    """Build the plant position schema."""
    d = defaults or {}
    return vol.Schema(
        {
            vol.Required(
                CONF_PLANT_DIST_WALL1,
                default=d.get(CONF_PLANT_DIST_WALL1, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=100.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_PLANT_DIST_WALL2,
                default=d.get(CONF_PLANT_DIST_WALL2, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=100.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_PLANT_RADIUS,
                default=d.get(CONF_PLANT_RADIUS, 0.3),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0.01, max=10.0, step=0.01, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_PLANT_Z_MIN,
                default=d.get(CONF_PLANT_Z_MIN, 0.0),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
            vol.Required(
                CONF_PLANT_Z_MAX,
                default=d.get(CONF_PLANT_Z_MAX, 1.2),
            ): NumberSelector(
                NumberSelectorConfig(
                    min=0, max=20.0, step=0.1, mode=NumberSelectorMode.BOX
                )
            ),
        }
    )


class SunShadeConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Sun Shade Integration."""

    VERSION = 2

    def __init__(self) -> None:
        """Initialize the config flow."""
        self._update_interval: int = DEFAULT_UPDATE_INTERVAL
        self._walls: list[dict] = []
        self._wall_defaults: list[dict] = []  # per-wall window defaults
        self._windows: list[dict] = []
        self._wall_counter: int = 0
        self._window_queue: list[tuple[str, int, dict]] = []  # (wall_id, count, defaults)
        self._current_window_index: int = 0

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 1: General settings."""
        if user_input is not None:
            self._update_interval = int(user_input[CONF_UPDATE_INTERVAL])
            self._wall_counter = 0
            return await self.async_step_wall()

        return self.async_show_form(
            step_id="user",
            data_schema=_user_schema(),
        )

    async def async_step_wall(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 2: Define a wall (loops with add_another_wall)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            wall_id = user_input[CONF_WALL_ID].strip()

            # Validate duplicate wall IDs
            if any(w["id"] == wall_id for w in self._walls):
                errors[CONF_WALL_ID] = "duplicate_wall_id"

            z_bottom = user_input[CONF_DEFAULT_Z_BOTTOM]
            z_top = user_input[CONF_DEFAULT_Z_TOP]
            if z_top <= z_bottom:
                errors[CONF_DEFAULT_Z_TOP] = "z_top_must_exceed_z_bottom"

            if not errors:
                self._walls.append(
                    {
                        "id": wall_id,
                        "outward_normal_azimuth_deg": float(user_input[CONF_OUTWARD_NORMAL]),
                        "thickness": float(user_input[CONF_WALL_THICKNESS]),
                        "axis": user_input[CONF_WALL_AXIS],
                    }
                )
                window_count = int(user_input[CONF_WINDOW_COUNT])
                self._wall_defaults.append(
                    {
                        "wall_id": wall_id,
                        "window_count": window_count,
                        "width": float(user_input[CONF_DEFAULT_WINDOW_WIDTH]),
                        "height": float(user_input[CONF_DEFAULT_WINDOW_HEIGHT]),
                        "z_bottom": float(z_bottom),
                        "z_top": float(z_top),
                    }
                )

                if user_input.get(CONF_ADD_ANOTHER_WALL, False):
                    self._wall_counter += 1
                    return await self.async_step_wall()

                # Build window queue from all walls
                self._window_queue = []
                for wd in self._wall_defaults:
                    self._window_queue.append(
                        (wd["wall_id"], wd["window_count"], wd)
                    )
                self._current_window_index = 0
                self._windows = []
                return await self.async_step_window()

        self._wall_counter += 1
        return self.async_show_form(
            step_id="wall",
            data_schema=_wall_schema(self._wall_counter),
            errors=errors,
            description_placeholders={
                "wall_number": str(self._wall_counter),
            },
        )

    async def async_step_window(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 3: Define each window (loops through all walls x window_count)."""
        errors: dict[str, str] = {}

        if user_input is not None:
            window_id = user_input[CONF_WINDOW_ID].strip()

            if any(w["id"] == window_id for w in self._windows):
                errors[CONF_WINDOW_ID] = "duplicate_window_id"

            z_bottom = user_input[CONF_Z_BOTTOM]
            z_top = user_input[CONF_Z_TOP]
            if z_top <= z_bottom:
                errors[CONF_Z_TOP] = "z_top_must_exceed_z_bottom"

            if not errors:
                # Determine which wall this window belongs to
                wall_id = self._get_current_wall_id()
                shade = user_input.get(CONF_SHADE_ENTITY_ID, "")

                self._windows.append(
                    {
                        "id": window_id,
                        "wall_id": wall_id,
                        "position_along_wall": float(user_input[CONF_POSITION_ALONG_WALL]),
                        "width": float(user_input[CONF_WINDOW_WIDTH]),
                        "height": float(user_input[CONF_WINDOW_HEIGHT]),
                        "z_bottom": float(z_bottom),
                        "z_top": float(z_top),
                        "shade_entity_id": shade if shade else None,
                    }
                )
                self._current_window_index += 1

                if self._current_window_index < self._total_windows():
                    return await self.async_step_window()
                return await self.async_step_plant()

        # Determine current wall and window number for description
        wall_id, win_num, defaults = self._get_current_window_info()
        letter = chr(ord("a") + win_num)
        suggested_id = f"{wall_id.replace('wall_', 'window_')}{letter}"

        window_defaults = {
            CONF_WINDOW_ID: suggested_id,
            CONF_WINDOW_WIDTH: defaults["width"],
            CONF_WINDOW_HEIGHT: defaults["height"],
            CONF_Z_BOTTOM: defaults["z_bottom"],
            CONF_Z_TOP: defaults["z_top"],
        }

        return self.async_show_form(
            step_id="window",
            data_schema=_window_schema(window_defaults),
            errors=errors,
            description_placeholders={
                "window_number": str(win_num + 1),
                "wall_id": wall_id,
                "wall_axis": self._get_wall_axis(wall_id),
            },
        )

    async def async_step_plant(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Step 4: Plant position."""
        if user_input is not None:
            await self.async_set_unique_id(DOMAIN)
            self._abort_if_unique_id_configured()

            return self.async_create_entry(
                title="Sun Shade Integration",
                data={
                    CONF_UPDATE_INTERVAL: self._update_interval,
                    CONF_WALLS: self._walls,
                    CONF_WINDOWS: self._windows,
                    CONF_PLANT: {
                        "dist_from_wall1": float(user_input[CONF_PLANT_DIST_WALL1]),
                        "dist_from_wall2": float(user_input[CONF_PLANT_DIST_WALL2]),
                        "radius": float(user_input[CONF_PLANT_RADIUS]),
                        "z_min": float(user_input[CONF_PLANT_Z_MIN]),
                        "z_max": float(user_input[CONF_PLANT_Z_MAX]),
                    },
                },
            )

        wall_1_id = self._walls[0]["id"] if self._walls else "wall_1"
        wall_2_id = self._walls[1]["id"] if len(self._walls) > 1 else "wall_2"

        return self.async_show_form(
            step_id="plant",
            data_schema=_plant_schema(),
            description_placeholders={
                "wall_1_id": wall_1_id,
                "wall_2_id": wall_2_id,
            },
        )

    def _total_windows(self) -> int:
        """Total number of windows across all walls."""
        return sum(wd["window_count"] for _, wd in [(None, d) for d in self._wall_defaults])

    def _get_current_wall_id(self) -> str:
        """Get the wall_id for the current window index."""
        idx = 0
        for wd in self._wall_defaults:
            idx += wd["window_count"]
            if self._current_window_index < idx:
                return wd["wall_id"]
        return self._wall_defaults[-1]["wall_id"]

    def _get_current_window_info(self) -> tuple[str, int, dict]:
        """Get (wall_id, window_number_within_wall, defaults) for current index."""
        idx = 0
        for wd in self._wall_defaults:
            if self._current_window_index < idx + wd["window_count"]:
                win_num = self._current_window_index - idx
                return wd["wall_id"], win_num, wd
            idx += wd["window_count"]
        wd = self._wall_defaults[-1]
        return wd["wall_id"], 0, wd

    def _get_wall_axis(self, wall_id: str) -> str:
        """Get the axis for a wall by ID."""
        for w in self._walls:
            if w["id"] == wall_id:
                return w["axis"].upper()
        return "?"

    @staticmethod
    @callback
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        """Get the options flow handler."""
        return SunShadeOptionsFlow(config_entry)


class SunShadeOptionsFlow(config_entries.OptionsFlow):
    """Menu-based options flow for viewing and editing configuration."""

    MENU_OPTIONS = ["opt_general", "opt_walls", "opt_windows", "opt_plant"]

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        """Initialize options flow."""
        self._config_entry = config_entry
        # Work on copies so we can mutate freely
        self._data: dict = dict(config_entry.data)
        self._walls: list[dict] = [dict(w) for w in self._data.get(CONF_WALLS, [])]
        self._windows: list[dict] = [dict(w) for w in self._data.get(CONF_WINDOWS, [])]
        self._plant: dict = dict(self._data.get(CONF_PLANT, {}))
        self._update_interval: int = self._data.get(CONF_UPDATE_INTERVAL, DEFAULT_UPDATE_INTERVAL)
        self._selected_wall_id: str | None = None
        self._selected_window_id: str | None = None

    def _save(self) -> None:
        """Persist current state back to the config entry."""
        self.hass.config_entries.async_update_entry(
            self._config_entry,
            data={
                CONF_UPDATE_INTERVAL: self._update_interval,
                CONF_WALLS: self._walls,
                CONF_WINDOWS: self._windows,
                CONF_PLANT: self._plant,
            },
        )

    # ── Main menu ──────────────────────────────────────────────

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Show the main options menu."""
        wall_summary = ", ".join(w["id"] for w in self._walls) or "none"
        window_summary = ", ".join(w["id"] for w in self._windows) or "none"

        menu_labels = {
            "opt_general": f"General settings (interval: {self._update_interval}s)",
            "opt_walls": f"Walls ({len(self._walls)}): {wall_summary}",
            "opt_windows": f"Windows ({len(self._windows)}): {window_summary}",
            "opt_plant": "Plant position",
        }

        return self.async_show_menu(
            step_id="init",
            menu_options=menu_labels,
        )

    # ── General settings ───────────────────────────────────────

    async def async_step_opt_general(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Edit general settings."""
        if user_input is not None:
            self._update_interval = int(user_input[CONF_UPDATE_INTERVAL])
            self._save()
            return await self.async_step_init()

        return self.async_show_form(
            step_id="opt_general",
            data_schema=_user_schema({CONF_UPDATE_INTERVAL: self._update_interval}),
        )

    # ── Walls: list / select ───────────────────────────────────

    async def async_step_opt_walls(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Show wall picker: select existing wall to edit, or add new."""
        if user_input is not None:
            choice = user_input.get("selected_wall", "")
            if choice == "__add_new__":
                self._selected_wall_id = None
                return await self.async_step_edit_wall()
            if choice:
                self._selected_wall_id = choice
                return await self.async_step_edit_wall()

        options = [
            SelectSelector(
                SelectSelectorConfig(
                    options=[
                        {"value": w["id"], "label": f"{w['id']} ({w['axis'].upper()}-axis, {w['outward_normal_azimuth_deg']}°)"}
                        for w in self._walls
                    ] + [{"value": "__add_new__", "label": "Add new wall..."}],
                    mode=SelectSelectorMode.LIST,
                )
            )
        ]

        return self.async_show_form(
            step_id="opt_walls",
            data_schema=vol.Schema(
                {vol.Required("selected_wall"): options[0]}
            ),
        )

    async def async_step_edit_wall(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Edit or add a single wall."""
        errors: dict[str, str] = {}

        if user_input is not None:
            wall_id = user_input[CONF_WALL_ID].strip()

            # Check duplicate (but allow keeping the same ID when editing)
            existing_ids = [w["id"] for w in self._walls if w["id"] != self._selected_wall_id]
            if wall_id in existing_ids:
                errors[CONF_WALL_ID] = "duplicate_wall_id"

            if not errors:
                new_wall = {
                    "id": wall_id,
                    "outward_normal_azimuth_deg": float(user_input[CONF_OUTWARD_NORMAL]),
                    "thickness": float(user_input[CONF_WALL_THICKNESS]),
                    "axis": user_input[CONF_WALL_AXIS],
                }

                if self._selected_wall_id:
                    # Update existing wall
                    old_id = self._selected_wall_id
                    for i, w in enumerate(self._walls):
                        if w["id"] == old_id:
                            self._walls[i] = new_wall
                            break
                    # Update wall_id references in windows if wall was renamed
                    if old_id != wall_id:
                        for win in self._windows:
                            if win.get("wall_id") == old_id:
                                win["wall_id"] = wall_id
                else:
                    # Add new wall
                    self._walls.append(new_wall)

                self._save()
                return await self.async_step_init()

        # Pre-populate defaults
        defaults: dict[str, Any] = {}
        wall_number = len(self._walls) + 1
        if self._selected_wall_id:
            for w in self._walls:
                if w["id"] == self._selected_wall_id:
                    defaults = {
                        CONF_WALL_ID: w["id"],
                        CONF_OUTWARD_NORMAL: w["outward_normal_azimuth_deg"],
                        CONF_WALL_THICKNESS: w["thickness"],
                        CONF_WALL_AXIS: w["axis"],
                    }
                    wall_number = self._walls.index(w) + 1
                    break
            # Fill window defaults from existing windows on this wall
            wall_windows = [win for win in self._windows if win.get("wall_id") == self._selected_wall_id]
            defaults[CONF_WINDOW_COUNT] = len(wall_windows) if wall_windows else 0
            if wall_windows:
                defaults[CONF_DEFAULT_WINDOW_WIDTH] = wall_windows[0]["width"]
                defaults[CONF_DEFAULT_WINDOW_HEIGHT] = wall_windows[0]["height"]
                defaults[CONF_DEFAULT_Z_BOTTOM] = wall_windows[0]["z_bottom"]
                defaults[CONF_DEFAULT_Z_TOP] = wall_windows[0]["z_top"]

        # Build schema without add_another_wall and window_count (not needed in edit mode)
        schema = _edit_wall_schema(wall_number, defaults)

        return self.async_show_form(
            step_id="edit_wall",
            data_schema=schema,
            errors=errors,
            description_placeholders={
                "wall_number": str(wall_number),
            },
        )

    # ── Windows: list / select ─────────────────────────────────

    async def async_step_opt_windows(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Show window picker: select existing window to edit, or add new."""
        if user_input is not None:
            choice = user_input.get("selected_window", "")
            if choice == "__add_new__":
                self._selected_window_id = None
                return await self.async_step_edit_window()
            if choice:
                self._selected_window_id = choice
                return await self.async_step_edit_window()

        options_list = []
        for win in self._windows:
            wall_axis = ""
            for w in self._walls:
                if w["id"] == win.get("wall_id"):
                    wall_axis = f", {w['axis'].upper()}-axis"
                    break
            shade = win.get("shade_entity_id") or "no shade"
            options_list.append({
                "value": win["id"],
                "label": f"{win['id']} on {win.get('wall_id', '?')}{wall_axis} — pos: {win['position_along_wall']}m, {shade}",
            })
        options_list.append({"value": "__add_new__", "label": "Add new window..."})

        return self.async_show_form(
            step_id="opt_windows",
            data_schema=vol.Schema(
                {
                    vol.Required("selected_window"): SelectSelector(
                        SelectSelectorConfig(
                            options=options_list,
                            mode=SelectSelectorMode.LIST,
                        )
                    )
                }
            ),
        )

    async def async_step_edit_window(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Edit or add a single window."""
        errors: dict[str, str] = {}

        if user_input is not None:
            window_id = user_input[CONF_WINDOW_ID].strip()
            wall_id = user_input.get("wall_id", "")

            existing_ids = [w["id"] for w in self._windows if w["id"] != self._selected_window_id]
            if window_id in existing_ids:
                errors[CONF_WINDOW_ID] = "duplicate_window_id"

            z_bottom = user_input[CONF_Z_BOTTOM]
            z_top = user_input[CONF_Z_TOP]
            if z_top <= z_bottom:
                errors[CONF_Z_TOP] = "z_top_must_exceed_z_bottom"

            if not errors:
                shade = user_input.get(CONF_SHADE_ENTITY_ID, "")
                new_window = {
                    "id": window_id,
                    "wall_id": wall_id,
                    "position_along_wall": float(user_input[CONF_POSITION_ALONG_WALL]),
                    "width": float(user_input[CONF_WINDOW_WIDTH]),
                    "height": float(user_input[CONF_WINDOW_HEIGHT]),
                    "z_bottom": float(z_bottom),
                    "z_top": float(z_top),
                    "shade_entity_id": shade if shade else None,
                }

                if self._selected_window_id:
                    for i, w in enumerate(self._windows):
                        if w["id"] == self._selected_window_id:
                            self._windows[i] = new_window
                            break
                else:
                    self._windows.append(new_window)

                self._save()
                return await self.async_step_init()

        # Pre-populate
        window_defaults: dict[str, Any] = {}
        wall_id_default = self._walls[0]["id"] if self._walls else ""
        if self._selected_window_id:
            for win in self._windows:
                if win["id"] == self._selected_window_id:
                    window_defaults = {
                        CONF_WINDOW_ID: win["id"],
                        CONF_POSITION_ALONG_WALL: win["position_along_wall"],
                        CONF_WINDOW_WIDTH: win["width"],
                        CONF_WINDOW_HEIGHT: win["height"],
                        CONF_Z_BOTTOM: win["z_bottom"],
                        CONF_Z_TOP: win["z_top"],
                        CONF_SHADE_ENTITY_ID: win.get("shade_entity_id", ""),
                    }
                    wall_id_default = win.get("wall_id", wall_id_default)
                    break

        # Get wall axis for description
        wall_axis = "?"
        for w in self._walls:
            if w["id"] == wall_id_default:
                wall_axis = w["axis"].upper()
                break

        # Build schema with wall_id selector
        wall_options = [w["id"] for w in self._walls]
        base_schema = _window_schema(window_defaults)
        schema_dict = dict(base_schema.schema)
        # Prepend wall_id selector
        new_schema_dict = {
            vol.Required("wall_id", default=wall_id_default): SelectSelector(
                SelectSelectorConfig(
                    options=wall_options,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
        }
        new_schema_dict.update(schema_dict)

        return self.async_show_form(
            step_id="edit_window",
            data_schema=vol.Schema(new_schema_dict),
            errors=errors,
            description_placeholders={
                "window_id": self._selected_window_id or "new",
                "wall_id": wall_id_default,
                "wall_axis": wall_axis,
            },
        )

    # ── Plant ──────────────────────────────────────────────────

    async def async_step_opt_plant(
        self, user_input: dict[str, Any] | None = None
    ) -> config_entries.ConfigFlowResult:
        """Edit plant position."""
        if user_input is not None:
            self._plant = {
                "dist_from_wall1": float(user_input[CONF_PLANT_DIST_WALL1]),
                "dist_from_wall2": float(user_input[CONF_PLANT_DIST_WALL2]),
                "radius": float(user_input[CONF_PLANT_RADIUS]),
                "z_min": float(user_input[CONF_PLANT_Z_MIN]),
                "z_max": float(user_input[CONF_PLANT_Z_MAX]),
            }
            self._save()
            return await self.async_step_init()

        plant_defaults = {
            CONF_PLANT_DIST_WALL1: self._plant.get("dist_from_wall1", 0.0),
            CONF_PLANT_DIST_WALL2: self._plant.get("dist_from_wall2", 0.0),
            CONF_PLANT_RADIUS: self._plant.get("radius", 0.3),
            CONF_PLANT_Z_MIN: self._plant.get("z_min", 0.0),
            CONF_PLANT_Z_MAX: self._plant.get("z_max", 1.2),
        }

        wall_1_id = self._walls[0]["id"] if self._walls else "wall_1"
        wall_2_id = self._walls[1]["id"] if len(self._walls) > 1 else "wall_2"

        return self.async_show_form(
            step_id="opt_plant",
            data_schema=_plant_schema(plant_defaults),
            description_placeholders={
                "wall_1_id": wall_1_id,
                "wall_2_id": wall_2_id,
            },
        )
