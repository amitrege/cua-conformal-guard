"""Adapters that map host CUA action formats into guard proposals."""

from cua_guard.adapters.base import ActionAdapter
from cua_guard.adapters.json_adapter import JsonActionAdapter
from cua_guard.adapters.osworld import OSWorldActionAdapter
from cua_guard.adapters.playwright import PlaywrightActionAdapter
from cua_guard.adapters.selenium import SeleniumActionAdapter

__all__ = [
    "ActionAdapter",
    "JsonActionAdapter",
    "OSWorldActionAdapter",
    "PlaywrightActionAdapter",
    "SeleniumActionAdapter",
]
