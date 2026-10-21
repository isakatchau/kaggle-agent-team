import numpy as np
import pandas as pd

from kagent.agents.schemas import FeaturePlan
from kagent.tools import metrics as M
from kagent.tools.features import build_features, validate_plan
from kagent.tools.kaggle import KaggleCLI, OfflineKaggle, percentile
from kagent.tools.profile import load_competition_files, profile
from kagent.tools.submission import check_submission, write_submission
from kagent.tools.train import blend, clip_params, run_cv


def test_metrics_direction():
    y = np.array([0, 1, 1, 0])
    assert M.score("roc_auc", y, np.array([0.1, 0.9, 0.8, 0.2])) == 1.0
    assert M.is_better("rmse", 0.1, 0.2) and not M.is_better("roc_auc", 0.1, 0.2)
    assert M.score("map@3", np.array([2]), np.array([[0.1, 0.5, 0.4]])) == 0.5


def test_percentile_respects_direction():
    board = [0.1, 0.2, 0.3, 0.4]
    assert percentile(0.35, board, "roc_auc") == 2 / 5   # one team ahead
    assert percentile(0.15, board, "rmse") == 2 / 5


def test_clip_params_bounds_and_unknowns():
    params, notes = clip_params("lightgbm", {"n_estimators": 10**6, "foo": 1, "learning_rate": "x"})
    assert params["n_estimators"] == 3000 and params["learning_rate"] == 0.03
    assert any("foo" in n for n in notes)


def test_validate_plan_drops_unknown_columns():
    plan = FeaturePlan(reasoning="", drop_columns=["nope", "a"], ratio_features=[("a", "b"), ("a", "zz")],
                       regex_features=[{"column": "t", "pattern": "(", "name": "bad"}])
    clean, issues = validate_plan(plan, ["a", "b", "t"], ["a", "b"])
    assert clean.drop_columns == ["a"] and clean.ratio_features == [("a", "b")]
    assert not clean.regex_features and len(issues) == 3


def test_features_train_submit_roundtrip(binary_comp, tmp_path):
    train, test, sample = load_competition_files(binary_comp)
    prof = profile(train, test, sample)
    assert prof["target_candidates"] == ["churned"]
    plan = FeaturePlan(reasoning="", datetime_columns=["signup_date"], drop_columns=["customer_code"],
                       regex_features=[{"column": "customer_code", "pattern": r"^(\w+)\.", "name": "title"}])
    fm = build_features(train, test, "churned", "id", plan)
    assert "title" in fm.categorical and "signup_date_month" in fm.X_train
    y = train["churned"].to_numpy()
    a = run_cv(fm, y, "binary", "roc_auc", "a", "lightgbm", {"n_estimators": 100}, 3, 0)
    b = run_cv(fm, y, "binary", "roc_auc", "b", "linear", {}, 3, 0)
    ens = blend([a, b], y, "roc_auc", "ens")
    assert ens.cv_score >= max(a.cv_score, b.cv_score) - 1e-9
    path = write_submission(tmp_path / "s.csv", sample, test["id"], ens.test_pred, "proba", np.array([0, 1]), "id")
    assert check_submission(path, sample) == []
    score = OfflineKaggle(binary_comp).submit("x", path, "test")
    assert 0.6 < score < 1.0


def test_label_submission_keeps_integer_dtype(tmp_path):
    sample = pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 0]})
    path = write_submission(tmp_path / "s.csv", sample, sample["PassengerId"], np.array([0.7, 0.2]),
                            "label", np.array([0, 1]), "PassengerId")
    assert pd.read_csv(path)["Survived"].tolist() == [1, 0]


def test_kaggle_submissions_csv_parsing():
    csv = "fileName,date,description,status,publicScore,privateScore\nsub.csv,2026,x,complete,0.78,\n"
    assert KaggleCLI._latest_public_score(csv) == 0.78
    assert KaggleCLI._latest_public_score("fileName,publicScore\nsub.csv,\n") is None
