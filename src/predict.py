"""Scoring and explanation with the saved Spark model (used by the web app and the CLI)."""

import json
import sys

import pandas as pd
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

from src import config
from src.spark_session import get_spark

REQUIRED = config.NUMERIC_COLS + config.CATEGORICAL_COLS
SCHEMA = StructType(
    [StructField(c, DoubleType()) for c in config.NUMERIC_COLS]
    + [StructField(c, StringType()) for c in config.CATEGORICAL_COLS]
    + [StructField("_row", LongType())]
)

# Factors checked when explaining one prediction, with the HR action each one suggests
DRIVERS = {
    "OverTime": "Review workload and overtime; rebalance shifts or add headcount.",
    "MonthlyIncome": "Benchmark pay against the market for this role and level.",
    "StockOptionLevel": "Consider stock options or a long-term retention bonus.",
    "JobLevel": "Discuss a career path and the next level with the employee.",
    "JobSatisfaction": "Hold a stay interview to find what makes the job unsatisfying.",
    "EnvironmentSatisfaction": "Look at team climate, workspace and manager relationship.",
    "WorkLifeBalance": "Offer flexible hours or remote days.",
    "JobInvolvement": "Give ownership of visible projects to raise engagement.",
    "RelationshipSatisfaction": "Check peer and manager relationships; consider mentoring.",
    "BusinessTravel": "Reduce or rotate travel commitments.",
    "DistanceFromHome": "Offer transport support, relocation help or hybrid work.",
    "YearsSinceLastPromotion": "Review promotion eligibility; overdue promotions drive exits.",
    "YearsWithCurrManager": "Strengthen the manager relationship; regular 1:1s.",
    "YearsAtCompany": "Early-tenure employees need structured onboarding and buddies.",
    "TotalWorkingYears": "Junior employees benefit from mentoring and skill plans.",
    "NumCompaniesWorked": "Frequent job changers: agree clear growth goals early.",
    "TrainingTimesLastYear": "Provide training and certification opportunities.",
    "MaritalStatus": "Not actionable; shown only as context.",
    "Age": "Not actionable; shown only as context.",
}


def validate(pdf: pd.DataFrame) -> list:
    """Return a list of problems with an uploaded file (empty list = OK)."""
    missing = [c for c in REQUIRED if c not in pdf.columns]
    problems = [f"Missing columns: {', '.join(missing)}"] if missing else []
    for c in config.NUMERIC_COLS:
        if c in pdf.columns and pd.to_numeric(pdf[c], errors="coerce").isna().any():
            problems.append(f"Column {c} has empty or non-numeric values")
    return problems


class AttritionPredictor:
    def __init__(self):
        self.spark = get_spark("EmployeeAttrition-Scoring")
        self.model = PipelineModel.load(str(config.BEST_MODEL_PATH))
        self.meta = json.loads(config.METRICS_PATH.read_text())
        self.threshold = self.meta["threshold"]
        stayers = pd.read_csv(config.DATA_PATH).query("Attrition == 'No'")
        # Typical value among employees who stayed: the "low-risk" reference for explanations
        self.reference = {
            c: (float(stayers[c].median()) if c in config.NUMERIC_COLS else stayers[c].mode()[0])
            for c in DRIVERS
        }

    def risk_band(self, p: float) -> str:
        if p >= self.threshold:
            return "High"
        if p >= 0.6 * self.threshold:
            return "Medium"
        return "Low"

    def _score(self, pdf: pd.DataFrame) -> pd.Series:
        rows = pdf[REQUIRED].copy()
        for c in config.NUMERIC_COLS:
            rows[c] = pd.to_numeric(rows[c]).astype(float)
        for c in config.CATEGORICAL_COLS:
            rows[c] = rows[c].astype(str)
        rows["_row"] = range(len(rows))
        sdf = self.spark.createDataFrame(rows[[f.name for f in SCHEMA.fields]], schema=SCHEMA)
        out = (self.model.transform(sdf)
               .select("_row", vector_to_array("probability")[1].alias("p"))
               .toPandas().sort_values("_row"))
        return pd.Series(out["p"].to_numpy(), index=pdf.index)

    def predict(self, pdf: pd.DataFrame) -> pd.DataFrame:
        result = pdf.copy()
        result["AttritionProbability"] = self._score(pdf).round(4)
        result["RiskBand"] = result["AttritionProbability"].map(self.risk_band)
        result["PredictedAttrition"] = (result["AttritionProbability"] >= self.threshold).map(
            {True: "Yes", False: "No"})
        return result

    def explain(self, record: dict, top: int = 6) -> pd.DataFrame:
        """What-if explanation: set one factor at a time to the typical stayer's value and
        measure how much the leave probability drops. A big drop = that factor drives risk."""
        variants = [dict(record)]
        names = []
        for factor, ref in self.reference.items():
            if record[factor] == ref:
                continue
            v = dict(record)
            v[factor] = ref
            variants.append(v)
            names.append(factor)
        probs = self._score(pd.DataFrame(variants)).to_numpy()
        base = probs[0]
        rows = [{"Factor": n, "Current": record[n], "Typical (stayed)": self.reference[n],
                 "Risk impact (pp)": round(100 * (base - p), 1), "Suggested action": DRIVERS[n]}
                for n, p in zip(names, probs[1:])]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        return df[df["Risk impact (pp)"] > 0.5].sort_values("Risk impact (pp)", ascending=False).head(top)


def main():
    """CLI: python -m src.predict input.csv [output.csv]"""
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict input.csv [output.csv]")
        sys.exit(1)
    pdf = pd.read_csv(sys.argv[1])
    problems = validate(pdf)
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    out = AttritionPredictor().predict(pdf)
    target = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"
    out.to_csv(target, index=False)
    print(out["RiskBand"].value_counts().to_string())
    print(f"Saved {len(out)} predictions to {target}")


if __name__ == "__main__":
    main()
