"""One place to build the local SparkSession (handles the Windows quirks)."""

import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession


def _windows_setup() -> None:
    # Spark's Python workers must use the same interpreter as the driver
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    if os.name != "nt":
        return
    if not os.environ.get("JAVA_HOME"):
        jdks = sorted(Path(r"C:\Program Files\Eclipse Adoptium").glob("jdk-17*"))
        if jdks:
            os.environ["JAVA_HOME"] = str(jdks[-1])
    if os.environ.get("HADOOP_HOME"):
        return
    # Saving models on Windows needs winutils.exe + hadoop.dll (see README)
    for candidate in (Path(__file__).resolve().parents[1] / "hadoop", Path.home() / "hadoop"):
        if (candidate / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(candidate)
            os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ["PATH"]
            return


def get_spark(app_name: str = "EmployeeAttrition") -> SparkSession:
    _windows_setup()
    spark = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "2g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
