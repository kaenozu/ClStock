import importlib
import sys

import pytest
from fastapi import HTTPException


@pytest.fixture(autouse=True)
def cleanup_security_state():
    import api.security as security

    security.reset_env_token_cache()
    yield
    security.reset_env_token_cache()


def test_security_imports_without_env_and_configures_roles(monkeypatch):
    target_module = "api.security"
    sys.modules.pop(target_module, None)
    sys.modules.pop("config.secrets", None)

    monkeypatch.delenv("CLSTOCK_DEV_KEY", raising=False)
    monkeypatch.delenv("CLSTOCK_ADMIN_KEY", raising=False)
    monkeypatch.delenv("API_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("API_USER_TOKEN", raising=False)
    monkeypatch.delenv("API_ENABLE_TEST_TOKENS", raising=False)

    security = importlib.import_module(target_module)

    # Placeholder keys should not provide privileged access
    with pytest.raises(HTTPException):
        security.verify_token(security.DUMMY_ADMIN_API_KEY)

    monkeypatch.setenv("CLSTOCK_DEV_KEY", "dev-real")
    monkeypatch.setenv("CLSTOCK_ADMIN_KEY", "admin-real")

    security.configure_security()

    assert security.verify_token("dev-real") == "developer"
    assert security.verify_token("admin-real") == "administrator"

    require_admin = security.require_role("admin")
    assert require_admin(user_type="administrator") == "administrator"
    with pytest.raises(HTTPException):
        require_admin(user_type="developer")
