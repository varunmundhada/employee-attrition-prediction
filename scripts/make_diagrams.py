"""Draws the architecture and pipeline diagrams used in the report and the slides."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config

BLUE, RED, GREEN, GOLD, GREY = "#2E6F8E", "#C8553D", "#5B8C5A", "#D9A441", "#6B6B6B"


def box(ax, x, y, w, h, title, body="", color=BLUE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color + "1A", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=10, weight="bold", color=color)
    if body:
        ax.text(x + w / 2, y + h - 0.48, body, ha="center", va="top", fontsize=7.6, color="#222222",
                linespacing=1.35)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=GREY, lw=1.4))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    box(ax, 0.2, 3.4, 2.0, 1.8, "Data layer", "IBM HR Attrition CSV\n1,470 employees\n35 attributes\n\nNew HR records\n(CSV upload / form)", GREY)
    box(ax, 2.7, 3.4, 2.6, 1.8, "Spark processing", "PySpark DataFrames\nSpark SQL analysis\nCleaning, de-duplication\nConstant-column removal\nStratified 85/15 split\nClass weights", BLUE)
    box(ax, 5.8, 3.4, 2.6, 1.8, "Spark MLlib", "Feature pipeline\n4 classifiers\n5-fold CrossValidator\nThreshold tuning\nBest PipelineModel saved", RED)
    box(ax, 8.9, 3.4, 1.9, 1.8, "Model store", "models/best_model\nmetrics.json\neda_summary.json\nfigures/*.png", GOLD)
    box(ax, 2.7, 0.6, 5.7, 1.7, "Streamlit web application", "HR Dashboard  ·  Predict an Employee  ·  Batch Prediction  ·  Model Performance\n\n"
        "AttritionPredictor loads the Spark PipelineModel once (cached),\nscores records, assigns Low / Medium / High risk bands\nand explains each prediction with what-if analysis", GREEN)
    box(ax, 8.9, 0.6, 1.9, 1.7, "HR user", "Views insights\nScores employees\nDownloads results\nPlans retention", GREY)
    arrow(ax, 2.2, 4.3, 2.7, 4.3)
    arrow(ax, 5.3, 4.3, 5.8, 4.3)
    arrow(ax, 8.4, 4.3, 8.9, 4.3)
    arrow(ax, 9.85, 3.4, 7.6, 2.3)
    arrow(ax, 8.4, 1.45, 8.9, 1.45)
    arrow(ax, 1.2, 3.4, 2.9, 2.3)
    ax.set_title("System architecture", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "00_architecture.png", dpi=160)
    plt.close(fig)


def pipeline():
    steps = [("SQLTransformer", "5 engineered\nfeatures"), ("StringIndexer", "7 categorical\ncolumns"),
             ("OneHotEncoder", "category\nvectors"), ("VectorAssembler", "one feature\nvector"),
             ("StandardScaler", "mean 0,\nstd 1"), ("Classifier", "LR / DT /\nRF / GBT")]
    fig, ax = plt.subplots(figsize=(11, 2.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2)
    ax.axis("off")
    w = 1.55
    for i, (t, b) in enumerate(steps):
        x = 0.15 + i * 1.8
        box(ax, x, 0.25, w, 1.5, t, b, RED if t == "Classifier" else BLUE)
        if i:
            arrow(ax, x - 0.25, 1.0, x, 1.0)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "00_ml_pipeline.png", dpi=160)
    plt.close(fig)


def methodology():
    steps = [("1. Business\nunderstanding", GREY), ("2. Data\ncollection", GREY), ("3. EDA with\nSpark SQL", BLUE),
             ("4. Cleaning &\nfeatures", BLUE), ("5. Model training\n& CV tuning", RED),
             ("6. Evaluation &\nselection", RED), ("7. Deployment\n(Streamlit)", GREEN), ("8. Insights &\nHR actions", GOLD)]
    fig, ax = plt.subplots(figsize=(11, 1.9))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 1.9)
    ax.axis("off")
    for i, (t, c) in enumerate(steps):
        x = 0.1 + i * 1.36
        ax.add_patch(FancyBboxPatch((x, 0.35), 1.15, 1.2, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=c + "1A", ec=c, lw=1.5))
        ax.text(x + 0.575, 0.95, t, ha="center", va="center", fontsize=8.2, weight="bold", color=c)
        if i:
            arrow(ax, x - 0.2, 0.95, x, 0.95)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "00_methodology.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    architecture()
    pipeline()
    methodology()
    print("Diagrams saved")
