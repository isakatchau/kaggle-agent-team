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
    # a new submission lands behind teams already at its score (board truncated to 5 decimals)
    assert percentile(328 / 418, [0.8, 0.78468, 0.78468, 0.7], "accuracy") == 4 / 5


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
    assert set(ens.params["weights"]) <= {"a", "b"} and len(ens.fold_scores) == 5
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


def test_blend_cv_is_not_inflated_by_weight_selection():
    # members with no signal: in-sample selection "finds" a gain over the best member, nested must not
    from kagent.tools import metrics as M
    from kagent.tools.train import ExperimentResult

    in_sample, nested = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, 400)
        oofs = [rng.random(400) for _ in range(8)]
        best = max(M.score("accuracy", y, o) for o in oofs)
        members = [ExperimentResult(name=f"m{i}", model="linear", params={}, cv_score=0.5, fold_scores=[0.5],
                                    oof=o, test_pred=rng.random(10), seconds=0) for i, o in enumerate(oofs)]
        ens = blend(members, y, "accuracy", "ens")
        in_sample.append(ens.params["in_sample_cv"] - best)
        nested.append(ens.cv_score - best)
    assert np.mean(in_sample) > 0.01
    assert np.mean(nested) <= 0


def _grouped(seed: int = 0, n_train: int = 900, n_test: int = 300):
    """Rows in groups of 1-4 that share a latent effect on y; `noise` is the same ids shuffled across rows."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    grp = np.repeat(np.arange(n), rng.integers(1, 5, n))[:n]
    rng.shuffle(grp)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = (x1 + rng.normal(0, 2.0, n)[grp] + rng.logistic(size=n) > 0).astype(int)
    df = pd.DataFrame({"id": np.arange(n), "x1": x1, "x2": x2, "grp": [f"g{g}" for g in grp],
                       "noise": [f"g{g}" for g in rng.permutation(grp)], "y": y})
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].drop(columns="y").reset_index(drop=True)


def test_group_target_encoding_never_reads_a_rows_own_label():
    from kagent.tools.features import TargetGroups, group_target_encode

    rng = np.random.default_rng(0)
    codes = rng.integers(-1, 40, 200)
    tg = TargetGroups("g", codes[:150], codes[150:])
    Y = rng.integers(0, 2, (150, 1)).astype(float)
    tr, va = np.arange(120), np.arange(120, 150)
    enc_tr, enc_va, enc_te = group_target_encode(tg, Y, tr, va)

    for i in (0, 5, 77):  # value = mean and count of the *other* training-fold members of the group
        others = [j for j in tr if codes[j] == codes[i] and j != i and codes[i] >= 0]
        assert enc_tr[i, 1] == len(others)
        assert np.isnan(enc_tr[i, 0]) if not others else np.isclose(enc_tr[i, 0], Y[others, 0].mean())
    flipped_va = Y.copy()
    flipped_va[va] = 1 - flipped_va[va]  # validation labels are never read
    for a, b in zip((enc_tr, enc_va, enc_te), group_target_encode(tg, flipped_va, tr, va)):
        assert np.array_equal(a, b, equal_nan=True)
    for i in range(0, 120, 7):  # flipping a training label never changes that row's own encoding
        flipped = Y.copy()
        flipped[i] = 1 - flipped[i]
        assert np.array_equal(group_target_encode(tg, flipped, tr, va)[0][i], enc_tr[i], equal_nan=True)


def test_smoothed_group_encoding_cannot_be_decoded_into_the_rows_own_label():
    # A training row's value must depend only on its group-mates' labels: two rows whose mates have the same
    # count and label sum get the same value whatever their own label. An exclude-self *prior* breaks this.
    from kagent.tools.features import TargetGroups, group_target_encode

    rng = np.random.default_rng(1)
    codes = rng.integers(0, 60, 300)
    Y = rng.integers(0, 2, (300, 1)).astype(float)
    tr, va = np.arange(250), np.arange(250, 300)
    enc_tr = group_target_encode(TargetGroups("g", codes, codes[:10], smoothing=3.0), Y, tr, va)[0]
    values: dict[tuple, set] = {}
    for i in tr:
        mates = (codes[tr] == codes[i]) & (tr != i)
        if mates.any():
            values.setdefault((int(mates.sum()), float(Y[tr][mates, 0].sum())), set()).add(enc_tr[i, 0])
    assert len(values) > 10 and all(len(v) == 1 for v in values.values())


def test_group_target_encoding_noise_keys_give_no_cv_gain():
    from kagent.agents.schemas import GroupTargetEncoding

    # one data seed moves accuracy by about +-0.01, so gains are averaged over three
    gains = {m: {"real": [], "noise": [], "leaky": []} for m in ("lightgbm", "linear")}
    for seed in range(3):
        train, test = _grouped(seed)
        y = train["y"].to_numpy()

        def features(key=None):
            enc = [GroupTargetEncoding(keys=[key], name="gte")] if key else []
            return build_features(train, test, "y", "id",
                                  FeaturePlan(reasoning="", drop_columns=["grp", "noise"], group_target_encoding=enc))

        leaky = features()  # control: an encoding that includes the row's own label inflates CV even on noise keys
        leaky.X_train["naive"] = train.groupby("noise")["y"].transform("mean").to_numpy()
        leaky.X_test["naive"] = test["noise"].map(train.groupby("noise")["y"].mean()).to_numpy()
        for model, g in gains.items():
            def cv(fm):
                return run_cv(fm, y, "binary", "accuracy", "m", model, {"n_estimators": 100}, 5, 0).cv_score
            base = cv(features())
            g["real"].append(cv(features("grp")) - base)
            g["noise"].append(cv(features("noise")) - base)
            g["leaky"].append(cv(leaky) - base)
    for model, g in gains.items():
        assert np.mean(g["real"]) > 0.04, (model, g)    # true groups carry signal
        assert np.mean(g["noise"]) < 0.01, (model, g)   # shuffled keys, same cardinality: no gain
        assert np.mean(g["leaky"]) > 0.1, (model, g)    # the test does catch an encoding that leaks


def test_group_target_encoding_plan_validation_and_multiclass():
    from kagent.agents.schemas import GroupTargetEncoding

    plan = FeaturePlan(reasoning="", regex_features=[{"column": "t", "pattern": r"^(\w)", "name": "t0"}],
                       drop_columns=["t0"], group_target_encoding=[
                           GroupTargetEncoding(keys=["t0", "a"], name="ok"),
                           GroupTargetEncoding(keys=["missing"], name="bad_key"),
                           GroupTargetEncoding(keys=["a"], name="a")])  # name collides with a column
    clean, issues = validate_plan(plan, ["a", "t"], ["a"])
    assert [g.name for g in clean.group_target_encoding] == ["ok"] and clean.drop_columns == ["t0"]
    assert len(issues) == 2

    # the regex does not match "-": that row belongs to no group; sizes count train+test rows
    train = pd.DataFrame({"id": range(6), "t": ["xa", "xb", "ya", "yb", "xc", "-"], "a": [1, 1, 2, 2, 1, 1],
                          "c": [0, 1, 2, 2, 0, 1]})
    test = train.drop(columns="c").assign(id=range(6, 12))
    fm = build_features(train, test, "c", "id", clean)
    assert "t0" not in fm.X_train and fm.X_train["ok_size"].tolist() == [6, 6, 4, 4, 6, 1]
    assert fm.target_groups[0].train[5] == -1
    r = run_cv(fm, train["c"].to_numpy(), "multiclass", "log_loss", "m", "linear", {}, 2, 0)
    assert r.oof.shape == (6, 3)
    assert {"ok_0", "ok_1", "ok_2", "ok_n"} <= set(fm.target_groups[0].columns(3))


def test_repeated_cv_scores_every_fold_of_every_repeat(binary_comp):
    train, test, sample = load_competition_files(binary_comp)
    fm = build_features(train, test, "churned", "id", FeaturePlan(reasoning=""))
    y = train["churned"].to_numpy()
    r = run_cv(fm, y, "binary", "accuracy", "lin", "linear", {}, 3, 0, repeats=2)
    assert len(r.fold_scores) == 6 and r.oof.min() >= 0 and r.oof.max() <= 1
    assert r.summary()["cv_lcb"] == round(r.cv_score - r.cv_se, 6)


def test_group_target_encoding_output_names_cannot_collide_with_generated_columns():
    # an LLM answer like this passed validation, then every LightGBM fit failed on a duplicate feature name
    plan = FeaturePlan(reasoning="", frequency_encode_columns=["booking"],
                       group_target_encoding=[{"keys": ["booking"], "name": "booking_freq"},
                                              {"keys": ["booking"], "name": "grp"},
                                              {"keys": ["booking"], "name": "grp_n"}])
    clean, issues = validate_plan(plan, ["booking", "x"], ["x"])
    assert [g.name for g in clean.group_target_encoding] == ["grp"]
    assert len(issues) == 2


def test_group_target_encoding_skips_category_keys_with_big_groups():
    # a 3-value category is not a group of related rows: LOO means inside it encode each row's label
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"id": range(200), "cat": rng.choice(list("abc"), 200),
                          "booking": [f"b{i // 2}" for i in range(200)], "y": rng.integers(0, 2, 200)})
    test = train.drop(columns="y").assign(id=range(200, 400))
    plan = FeaturePlan(reasoning="", group_target_encoding=[{"keys": ["cat"], "name": "cat_grp"},
                                                            {"keys": ["booking"], "name": "bk_grp"}])
    fm = build_features(train, test, "y", "id", plan)
    assert [t.name for t in fm.target_groups] == ["bk_grp"]
    assert len(fm.skipped) == 1 and "cat" in fm.skipped[0] and "cat_grp_size" not in fm.X_train


def _households(seed: int, shuffle_keys: bool = False):
    # members of a household share the outcome; a weak row feature explains a little of it
    rng = np.random.default_rng(seed)
    hh = np.repeat(np.arange(300), 3)
    fate = rng.integers(0, 2, 300)[hh]
    x = fate + rng.normal(0, 1.5, len(hh))
    y = np.where(rng.random(len(hh)) < 0.9, fate, 1 - fate)
    key = np.array([f"h{h}" for h in hh])
    if shuffle_keys:
        key = rng.permutation(key)
    df = pd.DataFrame({"id": range(len(hh)), "x": x, "key": key, "y": y})
    train, test = df.iloc[:600].reset_index(drop=True), df.iloc[600:].drop(columns="y").reset_index(drop=True)
    plan = FeaturePlan(reasoning="", drop_columns=["key"], group_target_encoding=[{"keys": ["key"], "name": "hh"}])
    return build_features(train, test, "y", "id", plan), train["y"].to_numpy()


def _no_groups(fm):
    # the same matrix with every row outside any group: group_override reduces to its tree
    import dataclasses
    from kagent.tools.features import TargetGroups

    empty = [TargetGroups(g.name, np.full_like(g.train, -1), np.full_like(g.test, -1)) for g in fm.target_groups]
    return dataclasses.replace(fm, target_groups=empty)


def test_group_override_uses_unanimous_groups_and_ignores_shuffled_keys():
    gains, noise = [], []
    for seed in range(3):
        fm, y = _households(seed)
        stump = run_cv(_no_groups(fm), y, "binary", "accuracy", "s", "group_override", {"depth": 1}, 5, 0)
        over = run_cv(fm, y, "binary", "accuracy", "o", "group_override", {"depth": 1}, 5, 0)
        fm_n, y_n = _households(seed, shuffle_keys=True)
        stump_n = run_cv(_no_groups(fm_n), y_n, "binary", "accuracy", "s", "group_override", {"depth": 1}, 5, 0)
        over_n = run_cv(fm_n, y_n, "binary", "accuracy", "o", "group_override", {"depth": 1}, 5, 0)
        gains.append(over.cv_score - stump.cv_score)
        noise.append(over_n.cv_score - stump_n.cv_score)
    assert np.mean(gains) > 0.05
    assert abs(np.mean(noise)) < 0.01


def test_group_override_never_reads_a_validation_or_own_label():
    from kagent.tools.train import _unanimous

    fm, y = _households(0)
    tg = fm.target_groups
    tr, va = np.arange(0, 480), np.arange(480, 600)
    flipped = y.copy()
    flipped[va] = 1 - flipped[va]
    a = _unanimous(tg, y, tr, [g.train[va] for g in tg], 1)
    b = _unanimous(tg, flipped, tr, [g.train[va] for g in tg], 1)
    assert (a == b).all()
    own = _unanimous(tg, y, tr, [g.train[tr] for g in tg], 1, own=y[tr])
    one = y.copy()
    one[5] = 1 - one[5]  # a training row's own label must not change its own verdict
    own2 = _unanimous(tg, one, tr, [g.train[tr] for g in tg], 1, own=one[tr])
    assert own[5] == own2[5]


def test_group_override_with_shuffled_labels_stays_at_the_majority_rate():
    # category ordering and group verdicts both use labels: with labels that carry no signal, CV must not rise
    rng = np.random.default_rng(1)
    n = 600
    train = pd.DataFrame({"id": range(n), "cat": rng.choice(list("abcdefgh"), n), "x": rng.normal(size=n),
                          "key": [f"k{i // 3}" for i in range(n)], "y": rng.integers(0, 2, n)})
    test = train.drop(columns="y").assign(id=range(n, 2 * n))
    plan = FeaturePlan(reasoning="", drop_columns=["key"], group_target_encoding=[{"keys": ["key"], "name": "g"}])
    fm = build_features(train, test, "y", "id", plan)
    y = train["y"].to_numpy()
    scores = [run_cv(fm, rng.permutation(y), "binary", "accuracy", "o", "group_override", {}, 5, s).cv_score
              for s in range(3)]
    assert np.mean(scores) < max(y.mean(), 1 - y.mean()) + 0.02


def test_group_override_ignores_rare_category_levels_when_ordering():
    # a categorical unique per row would hand the tree each row's own label if levels were ranked by
    # their own rows; rare levels get the prior, so the tree splits on the real feature instead
    rng = np.random.default_rng(3)
    n = 500
    x = rng.integers(0, 2, n)
    y = np.where(rng.random(n) < 0.85, x, 1 - x)
    train = pd.DataFrame({"id": range(n), "x": x, "uid": [f"u{i}" for i in range(n)],
                          "key": [f"k{i // 2}" for i in range(n)], "y": y})
    test = train.drop(columns="y").assign(id=range(n, 2 * n), uid=[f"t{i}" for i in range(n)])
    plan = FeaturePlan(reasoning="", drop_columns=["key"], group_target_encoding=[{"keys": ["key"], "name": "g"}])
    fm = build_features(train, test, "y", "id", plan)
    fm.categorical.append("uid") if "uid" not in fm.categorical else None
    r = run_cv(_no_groups(fm), y, "binary", "accuracy", "t", "group_override", {"depth": 1}, 5, 0)
    assert r.cv_score > 0.8


def test_cli_csv_skips_warnings_before_the_header():
    from kagent.tools.kaggle import KaggleCLI, csv_from_cli

    out = ("Warning: Looks like you're using an outdated `kaggle` version\n"
           "ref,fileName,date,description,status,publicScore,privateScore\n"
           "1,a.csv,2026-10-23,x,SubmissionStatus.COMPLETE,0.79426,\n")
    assert KaggleCLI._latest_public_score(out) == 0.79426
    assert list(csv_from_cli(out, "ref").columns)[:2] == ["ref", "fileName"]
    assert csv_from_cli("no csv here", "ref").empty
