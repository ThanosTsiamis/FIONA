import importlib.util
import json
import os
import sys
import types
from contextlib import contextmanager


ROOT = os.path.dirname(os.path.dirname(__file__))


def _stub_cachetools():
    module = types.ModuleType("cachetools")

    def cached(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    class TTLCache(dict):
        def __init__(self, *args, **kwargs):
            super().__init__()

    module.cached = cached
    module.TTLCache = TTLCache
    return module


def _stub_joblib():
    module = types.ModuleType("joblib")

    @contextmanager
    def parallel_backend(*args, **kwargs):
        yield

    class Parallel:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, iterable):
            return list(iterable)

    def delayed(func):
        return func

    module.parallel_backend = parallel_backend
    module.Parallel = Parallel
    module.delayed = delayed
    return module


def _stub_numpy():
    module = types.ModuleType("numpy")
    module.ndarray = list
    module.vectorize = lambda func: func
    module.asarray = lambda value: value
    module.unique = lambda values, return_counts=False: ([], []) if return_counts else []
    module.empty = lambda *args, **kwargs: []
    module.nonzero = lambda values: []
    module.isin = lambda *args, **kwargs: []
    module.where = lambda *args, **kwargs: []
    module.delete = lambda array, positions, axis=0: array
    module.argmax = lambda values: 0
    module.array = lambda value: value
    module.linspace = lambda start, stop, num: []
    module.percentile = lambda values, threshold: 0
    module.argwhere = lambda values: []
    module.ma = types.SimpleNamespace(
        median=lambda *args, **kwargs: types.SimpleNamespace(data=[]),
        masked_invalid=lambda *args, **kwargs: []
    )
    return module


def _stub_pandas():
    module = types.ModuleType("pandas")

    class DataFrame:
        pass

    class Series:
        def to_frame(self):
            return self

    module.DataFrame = DataFrame
    module.Series = Series
    module.read_csv = lambda *args, **kwargs: None
    module.read_excel = lambda *args, **kwargs: None
    module.read_json = lambda *args, **kwargs: None
    return module


def _stub_anytree():
    module = types.ModuleType("anytree")

    class Node:
        def __init__(self, name, children=None, parent=None, **kwargs):
            self.name = name
            self.children = children or []
            self.parent = parent
            self.data = kwargs.get("data", [])
            self.specificity_level = kwargs.get("specificity_level", 0)
            self.depth = kwargs.get("depth", 0)

        @property
        def leaves(self):
            if not self.children:
                return [self]
            result = []
            for child in self.children:
                result.extend(child.leaves)
            return result

    module.Node = Node
    return module


def _stub_doubledouble():
    module = types.ModuleType("doubledouble")

    class DoubleDouble(float):
        pass

    module.DoubleDouble = DoubleDouble
    return module


def _stub_flask():
    module = types.ModuleType("flask")

    class Headers(dict):
        def add(self, key, value):
            self[key] = value

    class Response:
        def __init__(self, response=None, content_type=None, status_code=200):
            self.response = response
            self.content_type = content_type
            self.status_code = status_code
            self.headers = Headers()

    class Flask:
        def __init__(self, name):
            self.name = name
            self.config = {}

        def route(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

        def errorhandler(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    def jsonify(data):
        response = Response(data, content_type="application/json")
        return response

    def redirect(location):
        response = Response(None, content_type=None)
        response.location = location
        return response

    module.Flask = Flask
    module.Response = Response
    module.jsonify = jsonify
    module.redirect = redirect
    module.request = types.SimpleNamespace(method="GET", files={}, form={})
    module.current_app = types.SimpleNamespace(config={})
    return module


def _stub_ujson():
    module = types.ModuleType("ujson")
    module.load = json.load
    module.dumps = json.dumps
    return module


def stubbed_modules():
    return {
        "cachetools": _stub_cachetools(),
        "joblib": _stub_joblib(),
        "numpy": _stub_numpy(),
        "pandas": _stub_pandas(),
        "anytree": _stub_anytree(),
        "chardet": types.SimpleNamespace(detect=lambda _: {"encoding": "utf-8"}),
        "doubledouble": _stub_doubledouble(),
        "psutil": types.ModuleType("psutil"),
        "flask": _stub_flask(),
        "ujson": _stub_ujson(),
    }


def load_module(module_name, relative_path, extra_modules=None):
    modules = stubbed_modules()
    if extra_modules:
        modules.update(extra_modules)

    previous = {name: sys.modules.get(name) for name in modules}
    previous_sys_path = list(sys.path)
    try:
        sys.modules.update(modules)
        module_path = os.path.join(ROOT, relative_path)
        module_dir = os.path.dirname(module_path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = previous_sys_path
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
