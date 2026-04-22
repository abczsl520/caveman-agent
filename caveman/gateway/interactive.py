"""Interactive — buttons, forms, and selection UI for chat platforms.

Provides platform-agnostic interactive components that render as
inline buttons (Discord), reply keyboards (Telegram), or text menus
(fallback). Extracted from OpenClaw src/interactive/ (308 lines).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

__all__ = [
    "Button",
    "ButtonRow",
    "SelectOption",
    "SelectMenu",
    "FormField",
    "Form",
    "InteractiveMessage",
    "InteractionEvent",
    "render_text_fallback",
    "render_discord_components",
    "ButtonStyle",
    "to_discord_components",
    "to_telegram_keyboard",
]


logger = logging.getLogger("caveman.gateway.interactive")


@dataclass
class Button:
    """A clickable button."""
    label: str
    value: str = ""
    style: str = "default"  # default | primary | danger | link
    url: str = ""  # For link buttons
    emoji: str = ""

    def __post_init__(self):
        if not self.value:
            self.value = self.label


@dataclass
class ButtonRow:
    """A row of buttons (max 5 per row on Discord)."""
    buttons: List[Button] = field(default_factory=list)

    def add(self, label: str, value: str = "", **kwargs) -> "ButtonRow":
        self.buttons.append(Button(label=label, value=value, **kwargs))
        return self


@dataclass
class SelectOption:
    """An option in a select menu."""
    label: str
    value: str
    description: str = ""
    emoji: str = ""


@dataclass
class SelectMenu:
    """A dropdown select menu."""
    placeholder: str = "Choose..."
    options: List[SelectOption] = field(default_factory=list)
    min_values: int = 1
    max_values: int = 1

    def add(self, label: str, value: str, description: str = "") -> "SelectMenu":
        self.options.append(SelectOption(label=label, value=value, description=description))
        return self

    def add_option(self, label: str, value: str, description: str = "") -> "SelectMenu":
        """Alias for add() (backward compat)."""
        return self.add(label, value, description)


@dataclass
class FormField:
    """A form input field."""
    name: str
    label: str
    placeholder: str = ""
    required: bool = True
    style: str = "short"  # short | paragraph
    default: str = ""
    min_length: int = 0
    max_length: int = 4000


@dataclass
class Form:
    """A modal form (Discord modal, Telegram inline form)."""
    title: str
    fields: List[FormField] = field(default_factory=list)
    submit_label: str = "Submit"

    def add_field(self, name: str, label: str, **kwargs) -> "Form":
        self.fields.append(FormField(name=name, label=label, **kwargs))
        return self


@dataclass
class InteractiveMessage:
    """A message with interactive components."""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    text: str = ""  # Alias for content (backward compat)
    button_rows: List[ButtonRow] = field(default_factory=list)
    select_menus: List[SelectMenu] = field(default_factory=list)
    ephemeral: bool = False  # Only visible to the user who triggered it

    def __post_init__(self):
        if self.text and not self.content:
            self.content = self.text

    def add_buttons(self, *buttons: Button) -> "InteractiveMessage":
        row = ButtonRow(buttons=list(buttons))
        self.button_rows.append(row)
        return self

    def add_select(self, menu_or_id=None, placeholder: str = "") -> SelectMenu:
        """Add a select menu. Accepts SelectMenu or (id, placeholder) for compat."""
        if isinstance(menu_or_id, SelectMenu):
            self.select_menus.append(menu_or_id)
            return menu_or_id
        menu = SelectMenu(placeholder=placeholder)
        self.select_menus.append(menu)
        return menu

    # ── Backward compat ──

    def add_row(self) -> "_CompatButtonRow":
        """Add a button row (backward compat API)."""
        row = ButtonRow()
        self.button_rows.append(row)
        return _CompatButtonRow(row)

    @property
    def rows(self) -> List[ButtonRow]:
        return self.button_rows

    @property
    def selects(self) -> List[SelectMenu]:
        return self.select_menus


@dataclass
class InteractionEvent:
    """An interaction from a user (button click, form submit, etc.)."""
    interaction_id: str
    message_id: str
    user_id: str
    component_type: str  # button | select | form
    value: str = ""
    values: List[str] = field(default_factory=list)
    form_data: Dict[str, str] = field(default_factory=dict)


# ── Rendering ──

def render_text_fallback(msg: InteractiveMessage) -> str:
    """Render interactive message as plain text (fallback for unsupported platforms)."""
    parts = []
    if msg.content:
        parts.append(msg.content)

    for row in msg.button_rows:
        options = " | ".join(f"[{b.label}]" for b in row.buttons)
        parts.append(f"Options: {options}")

    for menu in msg.select_menus:
        parts.append(f"\n{menu.placeholder}:")
        for i, opt in enumerate(menu.options, 1):
            desc = f" — {opt.description}" if opt.description else ""
            parts.append(f"  {i}. {opt.label}{desc}")

    return "\n".join(parts)


def render_discord_components(msg: InteractiveMessage) -> List[Dict[str, Any]]:
    """Render as Discord component payload."""
    components = []

    for row in msg.button_rows:
        buttons = []
        for btn in row.buttons:
            b: Dict[str, Any] = {
                "type": 2,  # Button
                "label": btn.label,
                "custom_id": f"{msg.id}:{btn.value}",
            }
            style_map = {"default": 2, "primary": 1, "danger": 4, "link": 5}
            # Auto-detect link style when url is present
            effective_style = "link" if btn.url else btn.style
            b["style"] = style_map.get(effective_style, 2)
            if btn.url:
                b["url"] = btn.url
                b.pop("custom_id", None)
            buttons.append(b)
        components.append({"type": 1, "components": buttons})

    for menu in msg.select_menus:
        options = [{
            "label": opt.label,
            "value": opt.value,
            "description": opt.description[:100] if opt.description else None,
        } for opt in menu.options]
        components.append({
            "type": 1,
            "components": [{
                "type": 3,  # Select menu
                "custom_id": f"{msg.id}:select",
                "placeholder": menu.placeholder,
                "options": options,
                "min_values": menu.min_values,
                "max_values": menu.max_values,
            }],
        })

    return components


# ── Backward Compatibility Layer ──

class _CompatButtonRow:
    """Wrapper for old API: row.add(label, action=, style=, url=)."""
    def __init__(self, row: ButtonRow):
        self._row = row

    def add(self, label: str, action: str = "", style: str = "default", url: str = "") -> "_CompatButtonRow":
        btn = Button(label=label, value=action or label, style=style, url=url)
        self._row.buttons.append(btn)
        return self

    @property
    def buttons(self) -> list:
        return self._row.buttons


class ButtonStyle:
    """Button style constants (backward compat)."""
    PRIMARY = "primary"
    SECONDARY = "default"
    SUCCESS = "primary"
    DANGER = "danger"
    LINK = "link"


# Aliases for old API
def to_discord_components(msg: "InteractiveMessage") -> List[Dict[str, Any]]:
    """Alias for render_discord_components."""
    return render_discord_components(msg)


def to_telegram_keyboard(msg: "InteractiveMessage") -> List[List[Dict[str, str]]]:
    """Render as Telegram inline keyboard."""
    rows = []
    for row in msg.button_rows:
        kb_row = []
        for btn in row.buttons:
            item: Dict[str, str] = {"text": btn.label}
            if btn.url:
                item["url"] = btn.url
            else:
                item["callback_data"] = btn.value
            kb_row.append(item)
        rows.append(kb_row)
    return rows
