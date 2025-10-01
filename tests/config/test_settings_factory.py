import importlib
import sys


def test_settings_lazy_loading(monkeypatch):
    monkeypatch.setenv("CLSTOCK_API_TITLE", "FromEnv")
    # Ensure we import a fresh module instance
    sys.modules.pop("config.settings", None)

    settings_module = importlib.import_module("config.settings")

    # The module should not eagerly create a settings singleton on import
    assert getattr(settings_module, "_SETTINGS_SINGLETON", None) is None

    # Explicitly creating settings should apply environment overrides
    created = settings_module.create_settings()
    assert created.api.title == "FromEnv"
