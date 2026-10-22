"""Paths and column definitions shared by the pipeline and the web app."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
MODEL_DIR = ROOT / "models"
BEST_MODEL_PATH = MODEL_DIR / "best_model"
METRICS_PATH = MODEL_DIR / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"

SEED = 42
LABEL = "label"
ID_COL = "EmployeeNumber"

# Columns that carry no information in this dataset (one value for every row)
CONSTANT_COLS = ["EmployeeCount", "Over18", "StandardHours"]

CATEGORICAL_COLS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

NUMERIC_COLS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Derived inside the Spark pipeline by a SQLTransformer, so raw HR records can be scored directly
ENGINEERED_COLS = [
    "IncomePerJobLevel",
    "TenureRatio",
    "YearsPerCompany",
    "PromotionGap",
    "SatisfactionIndex",
]

FEATURE_SQL = """
SELECT *,
       MonthlyIncome / JobLevel                                   AS IncomePerJobLevel,
       YearsAtCompany / (TotalWorkingYears + 1)                   AS TenureRatio,
       TotalWorkingYears / (NumCompaniesWorked + 1)               AS YearsPerCompany,
       YearsSinceLastPromotion / (YearsAtCompany + 1)             AS PromotionGap,
       (EnvironmentSatisfaction + JobSatisfaction + RelationshipSatisfaction
        + WorkLifeBalance + JobInvolvement) / 5.0                 AS SatisfactionIndex
FROM __THIS__
"""

# Allowed values for the categorical inputs in the web app
CATEGORY_VALUES = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": [
        "Human Resources", "Life Sciences", "Marketing", "Medical", "Other", "Technical Degree",
    ],
    "Gender": ["Female", "Male"],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician", "Manager",
        "Manufacturing Director", "Research Director", "Research Scientist",
        "Sales Executive", "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}

# Human-readable labels for the 1-4 / 1-5 rating scales in the dataset
SCALES = {
    "Education": {1: "Below College", 2: "College", 3: "Bachelor", 4: "Master", 5: "Doctor"},
    "EnvironmentSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobInvolvement": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "JobSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "PerformanceRating": {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"},
    "RelationshipSatisfaction": {1: "Low", 2: "Medium", 3: "High", 4: "Very High"},
    "WorkLifeBalance": {1: "Bad", 2: "Good", 3: "Better", 4: "Best"},
}
