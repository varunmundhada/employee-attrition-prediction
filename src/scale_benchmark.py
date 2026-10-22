"""Scalability test: grow the HR data synthetically and time the Spark pipeline at each size.

Each copy of the 1,470 records gets small random noise on its numeric columns, so the larger
datasets behave like a bigger workforce rather than exact duplicates. Usage:
    python -m src.scale_benchmark            # 1x, 10x, 50x, 200x, 500x
"""

import json
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F

from src import config
from src.preprocess import add_class_weights, build_pipeline, clean, load_raw
from src.spark_session import get_spark

NOISY = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate",
         "PercentSalaryHike", "TotalWorkingYears", "YearsAtCompany"]


def grow(df, factor):
    copies = df.crossJoin(df.sparkSession.range(factor).withColumnRenamed("id", "copy"))
    for c in NOISY:
        # +/- 5% noise, never below zero; copy 0 keeps the original values
        noise = F.when(F.col("copy") == 0, 1.0).otherwise(1 + (F.rand(config.SEED) - 0.5) * 0.1)
        copies = copies.withColumn(c, F.greatest(F.round(F.col(c) * noise), F.lit(0.0)))
    return copies.withColumn(config.ID_COL, F.col(config.ID_COL) + F.col("copy") * 1_000_000).drop("copy")


def run(factors=(1, 10, 50, 200, 500)):
    spark = get_spark("EmployeeAttrition-Scale")
    base = clean(load_raw(spark)).cache()
    evaluator = BinaryClassificationEvaluator(labelCol=config.LABEL)
    results = []
    for f in factors:
        df = add_class_weights(grow(base, f)).repartition(8).cache()
        rows = df.count()
        train, test = df.randomSplit([0.8, 0.2], seed=config.SEED)
        for name, clf in [
            ("Logistic Regression", LogisticRegression(labelCol=config.LABEL, weightCol="weight",
                                                       regParam=0.01, maxIter=50)),
            ("Random Forest", RandomForestClassifier(labelCol=config.LABEL, weightCol="weight",
                                                     numTrees=50, maxDepth=8, seed=config.SEED)),
        ]:
            start = time.time()
            model = build_pipeline(clf).fit(train)
            fit_s = time.time() - start
            start = time.time()
            auc = evaluator.evaluate(model.transform(test))
            score_s = time.time() - start
            results.append({"factor": f, "rows": rows, "model": name, "fit_seconds": round(fit_s, 1),
                            "score_seconds": round(score_s, 1), "auc": round(auc, 4)})
            print(results[-1])
        df.unpersist()

    (config.MODEL_DIR / "scale_benchmark.json").write_text(json.dumps(results, indent=2))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, color in [("Logistic Regression", "#2E6F8E"), ("Random Forest", "#5B8C5A")]:
        pts = [r for r in results if r["model"] == name]
        ax.plot([r["rows"] for r in pts], [r["fit_seconds"] for r in pts], "o-", color=color, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Rows (log scale)")
    ax.set_ylabel("Training time (s)")
    ax.set_title("Spark MLlib training time as the dataset grows")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "17_scalability.png", dpi=130)
    print("Saved", config.FIGURES_DIR / "17_scalability.png")
    spark.stop()


if __name__ == "__main__":
    run(tuple(int(x) for x in sys.argv[1:]) or (1, 10, 50, 200, 500))
