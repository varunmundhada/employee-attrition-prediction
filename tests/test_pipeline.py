"""Smoke tests: python -m pytest -q"""

import pandas as pd
import pytest

from src import config
from src.preprocess import clean, load_raw, stratified_split
from src.spark_session import get_spark


@pytest.fixture(scope="module")
def spark():
    s = get_spark("EmployeeAttrition-Tests")
    yield s


def test_clean_drops_constant_columns_and_adds_label(spark):
    df = clean(load_raw(spark))
    assert df.count() == 1470
    for c in config.CONSTANT_COLS:
        assert c not in df.columns
    assert set(r[0] for r in df.select(config.LABEL).distinct().collect()) == {0.0, 1.0}


def test_split_is_disjoint_and_stratified(spark):
    df = clean(load_raw(spark))
    dev, test = stratified_split(df)
    ids = [set(r[0] for r in d.select(config.ID_COL).collect()) for d in (dev, test)]
    assert not (ids[0] & ids[1])
    assert sum(len(i) for i in ids) == 1470
    rate = lambda d: d.filter(f"{config.LABEL} = 1").count() / d.count()
    assert abs(rate(dev) - rate(df)) < 0.03
    assert abs(rate(test) - rate(df)) < 0.03


@pytest.mark.skipif(not config.BEST_MODEL_PATH.exists(), reason="model not trained yet")
def test_predictor_scores_sample_file():
    from src.predict import AttritionPredictor, validate
    pdf = pd.read_csv(config.ROOT / "data" / "sample_employees.csv")
    assert validate(pdf) == []
    out = AttritionPredictor().predict(pdf)
    assert len(out) == len(pdf)
    assert out["AttritionProbability"].between(0, 1).all()
    assert set(out["RiskBand"]) <= {"Low", "Medium", "High"}


def test_validate_reports_missing_columns():
    from src.predict import validate
    problems = validate(pd.DataFrame({"Age": [30]}))
    assert problems and "Missing columns" in problems[0]
