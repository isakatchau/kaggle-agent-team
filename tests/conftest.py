from pathlib import Path

import pytest

from kagent.tools.synthetic import make_competition


@pytest.fixture(scope="session")
def binary_comp(tmp_path_factory) -> Path:
    return make_competition(tmp_path_factory.mktemp("data") / "synthetic-binary", "binary", n_train=800, n_test=400)


@pytest.fixture(scope="session")
def regression_comp(tmp_path_factory) -> Path:
    return make_competition(tmp_path_factory.mktemp("data") / "synthetic-regression", "regression",
                            n_train=800, n_test=400)
