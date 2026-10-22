"""Loading, cleaning and the shared feature pipeline."""

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    OneHotEncoder,
    SQLTransformer,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import Window
from pyspark.sql import functions as F

from src import config


def load_raw(spark: SparkSession, path=config.DATA_PATH) -> DataFrame:
    return spark.read.csv(str(path), header=True, inferSchema=True)


def clean(df: DataFrame) -> DataFrame:
    """Drop constant columns, remove duplicates and nulls, add the 0/1 label."""
    df = df.drop(*[c for c in config.CONSTANT_COLS if c in df.columns])
    df = df.dropDuplicates().dropna()
    if "Attrition" in df.columns:
        df = df.withColumn(config.LABEL, F.when(F.col("Attrition") == "Yes", 1.0).otherwise(0.0))
    for c in config.NUMERIC_COLS:
        df = df.withColumn(c, F.col(c).cast("double"))
    return df


def stratified_split(df: DataFrame, test_fraction=0.15, seed=config.SEED):
    """Development / test split that keeps the attrition rate equal in both parts.
    All tuning happens on the development set; the test set is used once at the end.

    Rows are ranked within each class by a seeded hash of the employee ID, so the split is
    exact (15% of leavers and 15% of stayers) and identical on every run and machine,
    unlike sampleBy, whose result depends on how Spark happens to partition the data."""
    rank = Window.partitionBy(config.LABEL).orderBy(F.xxhash64(F.col(config.ID_COL), F.lit(seed)))
    ranked = df.withColumn("_pct", F.percent_rank().over(rank))
    test = ranked.filter(F.col("_pct") < test_fraction).drop("_pct")
    dev = ranked.filter(F.col("_pct") >= test_fraction).drop("_pct")
    return dev, test


def add_class_weights(df: DataFrame) -> DataFrame:
    """Weight leavers up so both classes count equally (about 16% of employees leave)."""
    pos = df.filter(F.col(config.LABEL) == 1.0).count()
    neg = df.count() - pos
    w_pos = neg / pos
    return df.withColumn("weight", F.when(F.col(config.LABEL) == 1.0, w_pos).otherwise(1.0))


def feature_stages():
    """Stages that turn a raw HR record into a scaled feature vector."""
    engineer = SQLTransformer(statement=config.FEATURE_SQL)
    idx_cols = [f"{c}_idx" for c in config.CATEGORICAL_COLS]
    ohe_cols = [f"{c}_ohe" for c in config.CATEGORICAL_COLS]
    indexer = StringIndexer(
        inputCols=config.CATEGORICAL_COLS, outputCols=idx_cols, handleInvalid="keep"
    )
    encoder = OneHotEncoder(inputCols=idx_cols, outputCols=ohe_cols, handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=config.NUMERIC_COLS + config.ENGINEERED_COLS + ohe_cols,
        outputCol="raw_features",
        handleInvalid="keep",
    )
    scaler = StandardScaler(inputCol="raw_features", outputCol="features", withMean=True)
    return [engineer, indexer, encoder, assembler, scaler]


def feature_names(fitted_pipeline, sample: DataFrame) -> list:
    """Names of the slots in the assembled vector, in order, read from Spark's column
    metadata (one-hot columns appear as e.g. "OverTime = Yes")."""
    out = fitted_pipeline.transform(sample.limit(1))
    attrs = out.schema["raw_features"].metadata["ml_attr"]["attrs"]
    ordered = sorted((a for group in attrs.values() for a in group), key=lambda a: a["idx"])
    names = []
    for a in ordered:
        col, sep, value = a["name"].partition("_ohe_")
        names.append(f"{col} = {'unknown' if value == '__unknown' else value}" if sep else a["name"])
    return names


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(stages=feature_stages() + [classifier])
