"""Train and compare Spark MLlib classifiers, pick the best, save it with its metrics.

Steps: clean -> stratified 85/15 development / test split -> class weights ->
5-fold cross-validated grid search per model on the development set -> decision threshold
from out-of-fold predictions -> model with the best CV AUC selected -> one final evaluation
on the untouched test set -> save the best PipelineModel.
"""

import json
import shutil
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.classification import (
    DecisionTreeClassifier,
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

from src import config
from src.preprocess import add_class_weights, build_pipeline, clean, feature_names, load_raw, stratified_split
from src.spark_session import get_spark

COLORS = {"Logistic Regression": "#2E6F8E", "Decision Tree": "#8C8C8C",
          "Random Forest": "#5B8C5A", "Gradient-Boosted Trees": "#C8553D"}


def candidates():
    common = dict(featuresCol="features", labelCol=config.LABEL, weightCol="weight")
    lr = LogisticRegression(maxIter=100, **common)
    dt = DecisionTreeClassifier(seed=config.SEED, **common)
    rf = RandomForestClassifier(seed=config.SEED, **common)
    gbt = GBTClassifier(seed=config.SEED, stepSize=0.1, **common)
    return {
        "Logistic Regression": (lr, ParamGridBuilder()
                                .addGrid(lr.regParam, [0.001, 0.01, 0.1])
                                .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]).build()),
        "Decision Tree": (dt, ParamGridBuilder()
                          .addGrid(dt.maxDepth, [3, 5, 7])
                          .addGrid(dt.minInstancesPerNode, [5, 20]).build()),
        "Random Forest": (rf, ParamGridBuilder()
                          .addGrid(rf.numTrees, [100, 200])
                          .addGrid(rf.maxDepth, [5, 8]).build()),
        "Gradient-Boosted Trees": (gbt, ParamGridBuilder()
                                   .addGrid(gbt.maxIter, [50, 100])
                                   .addGrid(gbt.maxDepth, [3, 5]).build()),
    }


def scores(model, df):
    """(y_true, p_leave) as numpy arrays."""
    pdf = (model.transform(df)
           .select(config.LABEL, vector_to_array("probability")[1].alias("p"))
           .toPandas())
    return pdf[config.LABEL].to_numpy(), pdf["p"].to_numpy()


def metrics_at(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": round(float(threshold), 3), "accuracy": round((tp + tn) / len(y), 4),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def roc_points(y, p):
    order = np.argsort(-p)
    y = y[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr


def pr_points(y, p):
    order = np.argsort(-p)
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    recall = tp / y.sum()
    return recall, precision


def auc(y, p):
    fpr, tpr = roc_points(y, p)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # renamed in NumPy 2
    return float(trapezoid(tpr, fpr))


def oof_scores(clf, param_map, dev, k=5):
    """Out-of-fold probabilities on the development set for the tuned parameters:
    every employee is scored by a model that did not see them in training."""
    # Hash of the ID, not rand(), so the folds are the same on every run
    folds = dev.withColumn("_fold", F.pmod(F.xxhash64(F.col(config.ID_COL), F.lit(config.SEED + 1)), F.lit(k))).cache()
    ys, ps = [], []
    for i in range(k):
        model = build_pipeline(clf.copy(param_map)).fit(folds.filter(F.col("_fold") != i))
        y, p = scores(model, folds.filter(F.col("_fold") == i))
        ys.append(y)
        ps.append(p)
    folds.unpersist()
    return np.concatenate(ys), np.concatenate(ps)


def best_threshold(y, p):
    """Threshold that maximises F1 on the out-of-fold predictions."""
    grid = np.arange(0.10, 0.91, 0.01)
    f1s = [metrics_at(y, p, t)["f1"] for t in grid]
    return float(grid[int(np.argmax(f1s))])


def importances(model, sample):
    clf = model.stages[-1]
    names = feature_names(model, sample)
    if hasattr(clf, "featureImportances"):
        signed = clf.featureImportances.toArray()
    else:
        # Coefficient on standardised features: + raises the odds of leaving, - lowers them
        signed = clf.coefficients.toArray()
    pairs = sorted(zip(names, signed), key=lambda x: -abs(x[1]))
    return [{"feature": n, "importance": round(abs(float(v)), 5), "signed": round(float(v), 5)}
            for n, v in pairs]


def save_fig(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def run():
    spark = get_spark("EmployeeAttrition-Train")
    df = clean(load_raw(spark)).cache()
    dev, test = stratified_split(df)
    dev = add_class_weights(dev).cache()
    test = test.cache()
    split = {name: {"rows": d.count(), "leavers": d.filter(F.col(config.LABEL) == 1).count()}
             for name, d in [("development", dev), ("test", test)]}
    print("Split:", split)

    evaluator = BinaryClassificationEvaluator(labelCol=config.LABEL, metricName="areaUnderROC")
    results, fitted = {}, {}
    for name, (clf, grid) in candidates().items():
        start = time.time()
        cv = CrossValidator(estimator=build_pipeline(clf), estimatorParamMaps=grid,
                            evaluator=evaluator, numFolds=5, parallelism=4, seed=config.SEED)
        cv_model = cv.fit(dev)  # bestModel is refitted on the whole development set
        best = cv_model.bestModel
        best_map = grid[int(np.argmax(cv_model.avgMetrics))]
        best_params = {p.name: v for p, v in best_map.items()}
        yo, po = oof_scores(clf, best_map, dev)
        threshold = best_threshold(yo, po)
        yt, pt = scores(best, test)
        results[name] = {
            "cv_auc": round(float(max(cv_model.avgMetrics)), 4),
            "best_params": best_params,
            "oof_auc": round(auc(yo, po), 4),
            "oof_f1": metrics_at(yo, po, threshold)["f1"],
            "test_auc": round(auc(yt, pt), 4),
            "test_at_0.5": metrics_at(yt, pt, 0.5),
            "test": metrics_at(yt, pt, threshold),
            "train_seconds": round(time.time() - start, 1),
        }
        fitted[name] = (best, yt, pt)
        print(f"{name:24s} CV AUC {results[name]['cv_auc']:.3f}  OOF AUC {results[name]['oof_auc']:.3f}  "
              f"test AUC {results[name]['test_auc']:.3f}  test F1 {results[name]['test']['f1']:.3f}  "
              f"({results[name]['train_seconds']}s)")

    # Model selection uses the cross-validated AUC only; test stays an unbiased final check
    best_name = max(results, key=lambda n: results[n]["cv_auc"])
    best_model, yt, pt = fitted[best_name]
    print(f"Best model: {best_name}")

    if config.BEST_MODEL_PATH.exists():
        shutil.rmtree(config.BEST_MODEL_PATH)
    best_model.write().overwrite().save(str(config.BEST_MODEL_PATH))

    fpr, tpr = roc_points(yt, pt)
    keep = np.linspace(0, len(fpr) - 1, min(len(fpr), 120)).astype(int)
    payload = {
        "best_model": best_name,
        "threshold": results[best_name]["test"]["threshold"],
        "split": split,
        "models": results,
        "feature_importance": importances(best_model, dev)[:25],
        "roc_best": {"fpr": fpr[keep].round(4).tolist(), "tpr": tpr[keep].round(4).tolist()},
        "dataset_attrition_rate": round(100 * df.filter(F.col(config.LABEL) == 1).count() / df.count(), 2),
    }
    config.METRICS_PATH.write_text(json.dumps(payload, indent=2))

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, (_, y, p) in fitted.items():
        f, t = roc_points(y, p)
        ax.plot(f, t, color=COLORS[name], lw=1.8, label=f"{name} (AUC {results[name]['test_auc']:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="#BBBBBB", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves on the test set")
    ax.legend(fontsize=8, loc="lower right")
    save_fig(fig, "12_roc_curves")

    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for name, (_, y, p) in fitted.items():
        r, pr = pr_points(y, p)
        ax.plot(r, pr, color=COLORS[name], lw=1.8, label=name)
    ax.axhline(yt.mean(), ls="--", color="#BBBBBB", lw=1, label="Random baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves on the test set")
    ax.legend(fontsize=8)
    save_fig(fig, "13_pr_curves")

    names = list(results)
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, (metric, label) in enumerate([("auc", "AUC"), ("f1", "F1"), ("recall", "Recall"),
                                         ("precision", "Precision")]):
        vals = [results[n]["test_auc"] if metric == "auc" else results[n]["test"][metric] for n in names]
        ax.bar(x + (i - 1.5) * 0.2, vals, 0.2, label=label,
               color=["#2E6F8E", "#C8553D", "#5B8C5A", "#D9A441"][i])
    ax.set_xticks(x, [n.replace("Gradient-Boosted Trees", "GBT") for n in names])
    ax.set_ylim(0, 1)
    ax.set_title("Test-set metrics by model (tuned threshold)")
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    save_fig(fig, "14_model_comparison")

    c = results[best_name]["test"]["confusion"]
    cm = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted leave"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually left"])
    ax.set_title(f"Confusion matrix: {best_name}\n(threshold {results[best_name]['test']['threshold']})")
    save_fig(fig, "15_confusion_matrix")

    imp = payload["feature_importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    is_tree = hasattr(best_model.stages[-1], "featureImportances")
    ax.barh([d["feature"] for d in imp], [d["signed"] for d in imp],
            color=["#2E6F8E" if is_tree or d["signed"] < 0 else "#C8553D" for d in imp])
    ax.axvline(0, color="#888888", lw=0.8)
    ax.set_title(f"Top 15 features: {best_name}")
    ax.set_xlabel("Importance" if is_tree else
                  "Coefficient (standardised)\nred = raises risk · blue = lowers risk")
    save_fig(fig, "16_feature_importance")

    print(f"Saved model to {config.BEST_MODEL_PATH} and metrics to {config.METRICS_PATH}")
    spark.stop()


if __name__ == "__main__":
    run()
