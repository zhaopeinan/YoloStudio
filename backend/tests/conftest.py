from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.auth.deps import get_current_user
from app.main import app


TEST_ADMIN = SimpleNamespace(
    id=1,
    username="test-admin",
    role="admin",
    is_active=True,
    password_hash="unused",
)


@pytest.fixture(autouse=True)
def _bypass_auth_for_tests():
    app.dependency_overrides[get_current_user] = lambda: TEST_ADMIN
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_current_user, None)
