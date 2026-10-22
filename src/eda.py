"""Exploratory analysis with Spark SQL; saves charts to reports/figures and a JSON summary."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

from src import config
from src.preprocess import clean, load_raw
from src.spark_session import get_spark

STAY, LEAVE, ACCENT = "#2E6F8E", "#C8553D", "#8C8C8C"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 11,
})


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def rate_by(spark, column, order_by="rate DESC"):
    """Attrition rate (%) and headcount for each value of a column, via Spark SQL."""
    return spark.sql(f"""
        SELECT {column} AS grp, COUNT(*) AS n, ROUND(100 * AVG(label), 1) AS rate
        FROM hr GROUP BY {column} ORDER BY {order_by}
    """).toPandas()


def bar_rate(pdf, title, name, overall, horizontal=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.6 if not horizontal else 4.2))
    labels = pdf["grp"].astype(str)
    colors = [LEAVE if r > overall else STAY for r in pdf["rate"]]
    if horizontal:
        ax.barh(labels[::-1], pdf["rate"][::-1], color=colors[::-1])
        ax.axvline(overall, color=ACCENT, ls="--", lw=1)
        ax.set_xlabel("Attrition rate (%)")
        for i, (r, n) in enumerate(zip(pdf["rate"][::-1], pdf["n"][::-1])):
            ax.text(r + 0.4, i, f"{r:.1f}%  (n={n})", va="center", fontsize=8)
    else:
        ax.bar(labels, pdf["rate"], color=colors)
        ax.axhline(overall, color=ACCENT, ls="--", lw=1)
        ax.set_ylabel("Attrition rate (%)")
        for i, (r, n) in enumerate(zip(pdf["rate"], pdf["n"])):
            ax.text(i, r + 0.5, f"{r:.1f}%\n(n={n})", ha="center", fontsize=8)
    ax.set_title(title)
    _save(fig, name)


def run():
    spark = get_spark("EmployeeAttrition-EDA")
    df = clean(load_raw(spark)).cache()
    df.createOrReplaceTempView("hr")

    total = df.count()
    leavers = df.filter("label = 1").count()
    overall = round(100 * leavers / total, 1)
    print(f"Rows: {total}, columns after cleaning: {len(df.columns)}, attrition: {overall}%")

    # 1. Class balance
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.bar(["Stayed", "Left"], [total - leavers, leavers], color=[STAY, LEAVE])
    for i, v in enumerate([total - leavers, leavers]):
        ax.text(i, v + 15, f"{v} ({100 * v / total:.1f}%)", ha="center")
    ax.set_title("Employee attrition: class balance")
    ax.set_ylabel("Employees")
    _save(fig, "01_class_balance")

    # 2-6. Attrition rate by key factors
    spark.sql("""
        SELECT *,
          CASE WHEN Age < 25 THEN '18-24' WHEN Age < 35 THEN '25-34' WHEN Age < 45 THEN '35-44'
               WHEN Age < 55 THEN '45-54' ELSE '55+' END AS AgeBand,
          CASE WHEN MonthlyIncome < 3000 THEN '<3k' WHEN MonthlyIncome < 6000 THEN '3k-6k'
               WHEN MonthlyIncome < 10000 THEN '6k-10k' ELSE '10k+' END AS IncomeBand,
          CASE WHEN YearsAtCompany < 2 THEN '0-1' WHEN YearsAtCompany < 5 THEN '2-4'
               WHEN YearsAtCompany < 10 THEN '5-9' ELSE '10+' END AS TenureBand
        FROM hr
    """).createOrReplaceTempView("hr")

    tables = {
        "OverTime": rate_by(spark, "OverTime"),
        "JobRole": rate_by(spark, "JobRole"),
        "MaritalStatus": rate_by(spark, "MaritalStatus"),
        "BusinessTravel": rate_by(spark, "BusinessTravel"),
        "Department": rate_by(spark, "Department"),
        "AgeBand": rate_by(spark, "AgeBand", "grp"),
        "IncomeBand": rate_by(spark, "IncomeBand", "MIN(MonthlyIncome)"),
        "TenureBand": rate_by(spark, "TenureBand", "MIN(YearsAtCompany)"),
        "JobLevel": rate_by(spark, "JobLevel", "grp"),
        "JobSatisfaction": rate_by(spark, "JobSatisfaction", "grp"),
        "WorkLifeBalance": rate_by(spark, "WorkLifeBalance", "grp"),
        "StockOptionLevel": rate_by(spark, "StockOptionLevel", "grp"),
    }
    bar_rate(tables["OverTime"], "Attrition rate by overtime", "02_overtime", overall)
    bar_rate(tables["JobRole"], "Attrition rate by job role", "03_job_role", overall, horizontal=True)
    bar_rate(tables["IncomeBand"], "Attrition rate by monthly income band", "04_income", overall)
    bar_rate(tables["TenureBand"], "Attrition rate by years at company", "05_tenure", overall)
    bar_rate(tables["AgeBand"], "Attrition rate by age band", "06_age", overall)
    bar_rate(tables["MaritalStatus"], "Attrition rate by marital status", "07_marital", overall)
    bar_rate(tables["BusinessTravel"], "Attrition rate by business travel", "08_travel", overall)
    bar_rate(tables["StockOptionLevel"], "Attrition rate by stock option level", "09_stock", overall)

    # 10. Income distribution for stayers vs leavers
    inc = df.select("MonthlyIncome", "label").toPandas()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(inc.MonthlyIncome.min(), inc.MonthlyIncome.max(), 30)
    ax.hist(inc[inc.label == 0].MonthlyIncome, bins=bins, alpha=0.7, color=STAY, label="Stayed", density=True)
    ax.hist(inc[inc.label == 1].MonthlyIncome, bins=bins, alpha=0.7, color=LEAVE, label="Left", density=True)
    ax.set_xlabel("Monthly income")
    ax.set_ylabel("Density")
    ax.set_title("Monthly income: stayers vs leavers")
    ax.legend()
    _save(fig, "10_income_distribution")

    # 11. Correlation matrix with Spark MLlib
    corr_cols = ["label", "Age", "MonthlyIncome", "JobLevel", "TotalWorkingYears", "YearsAtCompany",
                 "YearsInCurrentRole", "YearsWithCurrManager", "YearsSinceLastPromotion",
                 "JobSatisfaction", "EnvironmentSatisfaction", "WorkLifeBalance", "StockOptionLevel",
                 "DistanceFromHome", "NumCompaniesWorked"]
    vec = VectorAssembler(inputCols=corr_cols, outputCol="v").transform(df).select("v")
    matrix = Correlation.corr(vec, "v").head()[0].toArray()
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = ["Attrition" if c == "label" else c for c in corr_cols]
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(matrix[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pearson correlation (Spark MLlib)")
    _save(fig, "11_correlation")

    summary = {
        "rows": total,
        "columns_raw": len(load_raw(spark).columns),
        "columns_clean": len(df.columns) - 1,  # minus the added label column
        "leavers": leavers,
        "attrition_rate": overall,
        "by": {k: v.to_dict(orient="records") for k, v in tables.items()},
        "corr_with_attrition": {
            labels[i]: round(float(matrix[0, i]), 3) for i in range(1, len(labels))
        },
    }
    config.MODEL_DIR.mkdir(exist_ok=True)
    (config.MODEL_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved charts to {config.FIGURES_DIR}")
    spark.stop()


if __name__ == "__main__":
    run()
