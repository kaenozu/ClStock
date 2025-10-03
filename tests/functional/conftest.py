import sys
import types
from types import SimpleNamespace

import pytest


def _create_fastapi_stubs():
    fastapi_stub = types.ModuleType("fastapi")

    class _FastAPI(SimpleNamespace):
        def mount(self, *_, **__):
            return None

        def get(self, *_, **__):
            def decorator(func):
                return func

            return decorator

        post = get
        put = get
        delete = get

    fastapi_stub.FastAPI = lambda *args, **kwargs: _FastAPI()
    fastapi_stub.Request = object

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.HTMLResponse = object

    templating_stub = types.ModuleType("fastapi.templating")

    class _Templates(SimpleNamespace):
        def __init__(self, *_, **__):
            super().__init__(env=SimpleNamespace(filters={}))

    templating_stub.Jinja2Templates = lambda *args, **kwargs: _Templates()

    staticfiles_stub = types.ModuleType("fastapi.staticfiles")
    staticfiles_stub.StaticFiles = lambda *args, **kwargs: SimpleNamespace()

    uvicorn_stub = types.ModuleType("uvicorn")
    numpy_stub = types.ModuleType("numpy")

    models_new_stub = types.ModuleType("models_new")
    models_new_stub.__path__ = []

    precision_pkg = types.ModuleType("models_new.precision")
    precision_pkg.__path__ = []

    precision_module = types.ModuleType("models_new.precision.precision_87_system")

    class _Precision87BreakthroughSystem(SimpleNamespace):
        def predict_with_87_precision(self, symbol: str):
            return {}

    precision_module.Precision87BreakthroughSystem = _Precision87BreakthroughSystem

    models_new_stub.precision = precision_pkg
    precision_pkg.precision_87_system = precision_module

    return {
        "fastapi": fastapi_stub,
        "fastapi.responses": responses_stub,
        "fastapi.templating": templating_stub,
        "fastapi.staticfiles": staticfiles_stub,
        "uvicorn": uvicorn_stub,
        "numpy": numpy_stub,
        "models_new": models_new_stub,
        "models_new.precision": precision_pkg,
        "models_new.precision.precision_87_system": precision_module,
    }


@pytest.fixture(autouse=True)
def fastapi_dependency_stubs():
    created_modules = _create_fastapi_stubs()
    original_modules = {name: sys.modules.get(name) for name in created_modules}

    for name, module in created_modules.items():
        if name not in sys.modules:
            sys.modules[name] = module

    try:
        yield
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
