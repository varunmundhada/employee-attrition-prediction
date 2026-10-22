"""Employee Attrition Prediction — HR analytics web app (Streamlit + Spark MLlib)."""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

st.set_page_config(page_title="Employee Attrition Prediction", page_icon="📉", layout="wide")

STAY, LEAVE = "#2E6F8E", "#C8553D"
BAND_COLORS = {"Low": "#5B8C5A", "Medium": "#D9A441", "High": "#C8553D"}


@st.cache_data
def load_data():
    return pd.read_csv(config.DATA_PATH)


@st.cache_data
def load_metrics():
    return json.loads(config.METRICS_PATH.read_text()) if config.METRICS_PATH.exists() else None


@st.cache_resource(show_spinner="Starting Spark and loading the model (first time only)…")
def load_predictor():
    from src.predict import AttritionPredictor
    return AttritionPredictor()


def rate_chart(df, col, title, horizontal=False):
    g = df.groupby(col)["Attrition"].apply(lambda s: round(100 * (s == "Yes").mean(), 1)).reset_index()
    g.columns = [col, "Attrition rate (%)"]
    g = g.sort_values("Attrition rate (%)", ascending=horizontal)
    fig = px.bar(g, x="Attrition rate (%)" if horizontal else col,
                 y=col if horizontal else "Attrition rate (%)",
                 orientation="h" if horizontal else "v", title=title, text="Attrition rate (%)",
                 color_discrete_sequence=[LEAVE])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    return fig


# --------------------------------------------------------------------------------------
st.sidebar.title("📉 Attrition Predictor")
st.sidebar.caption("HR Analytics · Apache Spark MLlib")
page = st.sidebar.radio("Go to", ["HR Dashboard", "Predict an Employee", "Batch Prediction",
                                  "Model Performance", "About"])
metrics = load_metrics()
if metrics is None:
    st.error("No trained model found. Run `python -m src.train` first.")
    st.stop()
st.sidebar.markdown("---")
st.sidebar.metric("Model in use", metrics["best_model"])
st.sidebar.metric("Test ROC-AUC", f"{metrics['models'][metrics['best_model']]['test_auc']:.3f}")
st.sidebar.caption(f"Decision threshold: {metrics['threshold']}")

data = load_data()

# --------------------------------------------------------------------------------------
if page == "HR Dashboard":
    st.title("HR Attrition Dashboard")
    st.caption("IBM HR Analytics Employee Attrition dataset · 1,470 employees")
    c1, c2, c3 = st.columns(3)
    depts = c1.multiselect("Department", sorted(data.Department.unique()))
    genders = c2.multiselect("Gender", sorted(data.Gender.unique()))
    ot = c3.multiselect("Overtime", ["Yes", "No"])
    df = data.copy()
    if depts:
        df = df[df.Department.isin(depts)]
    if genders:
        df = df[df.Gender.isin(genders)]
    if ot:
        df = df[df.OverTime.isin(ot)]
    if df.empty:
        st.warning("No employees match these filters.")
        st.stop()

    left = (df.Attrition == "Yes").sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Employees", f"{len(df):,}")
    k2.metric("Left the company", f"{left:,}")
    k3.metric("Attrition rate", f"{100 * left / len(df):.1f}%")
    k4.metric("Avg monthly income", f"{df.MonthlyIncome.mean():,.0f}")

    a, b = st.columns(2)
    a.plotly_chart(rate_chart(df, "OverTime", "Attrition by overtime"), use_container_width=True)
    b.plotly_chart(rate_chart(df, "JobRole", "Attrition by job role", horizontal=True),
                   use_container_width=True)
    a, b = st.columns(2)
    df = df.assign(TenureBand=pd.cut(df.YearsAtCompany, [-1, 1, 4, 9, 50],
                                     labels=["0-1 yrs", "2-4 yrs", "5-9 yrs", "10+ yrs"]).astype(str),
                   IncomeBand=pd.cut(df.MonthlyIncome, [0, 3000, 6000, 10000, 1e6],
                                     labels=["<3k", "3k-6k", "6k-10k", "10k+"]).astype(str))
    a.plotly_chart(rate_chart(df, "TenureBand", "Attrition by tenure"), use_container_width=True)
    b.plotly_chart(rate_chart(df, "IncomeBand", "Attrition by income band"), use_container_width=True)
    a, b = st.columns(2)
    a.plotly_chart(rate_chart(df, "MaritalStatus", "Attrition by marital status"),
                   use_container_width=True)
    a2 = px.histogram(df, x="Age", color="Attrition", barmode="overlay", nbins=30,
                      color_discrete_map={"No": STAY, "Yes": LEAVE}, title="Age distribution")
    a2.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))
    b.plotly_chart(a2, use_container_width=True)
    with st.expander("View data"):
        st.dataframe(df, use_container_width=True)

# --------------------------------------------------------------------------------------
elif page == "Predict an Employee":
    st.title("Predict attrition risk for one employee")
    st.caption("Enter the employee's HR record. The Spark model returns the probability of leaving "
               "and the factors driving it.")
    med = data.median(numeric_only=True)
    S = config.SCALES
    with st.form("employee"):
        st.subheader("Personal")
        c = st.columns(4)
        age = c[0].number_input("Age", 18, 60, 30)
        gender = c[1].selectbox("Gender", config.CATEGORY_VALUES["Gender"])
        marital = c[2].selectbox("Marital status", config.CATEGORY_VALUES["MaritalStatus"], index=2)
        distance = c[3].number_input("Distance from home (km)", 1, 30, 10)
        c = st.columns(4)
        education = c[0].selectbox("Education", list(S["Education"]), index=2,
                                   format_func=lambda v: S["Education"][v])
        edu_field = c[1].selectbox("Education field", config.CATEGORY_VALUES["EducationField"], index=1)
        num_companies = c[2].number_input("Companies worked at before", 0, 9, 1)
        total_years = c[3].number_input("Total working years", 0, 40, 5)

        st.subheader("Job")
        c = st.columns(4)
        dept = c[0].selectbox("Department", config.CATEGORY_VALUES["Department"], index=1)
        role = c[1].selectbox("Job role", config.CATEGORY_VALUES["JobRole"], index=6)
        level = c[2].selectbox("Job level", [1, 2, 3, 4, 5], index=0)
        travel = c[3].selectbox("Business travel", config.CATEGORY_VALUES["BusinessTravel"], index=1)
        c = st.columns(4)
        overtime = c[0].selectbox("Works overtime", ["No", "Yes"], index=1)
        training = c[1].number_input("Trainings last year", 0, 6, 2)
        perf = c[2].selectbox("Performance rating", [3, 4], format_func=lambda v: S["PerformanceRating"][v])
        involvement = c[3].selectbox("Job involvement", [1, 2, 3, 4], index=2,
                                     format_func=lambda v: S["JobInvolvement"][v])

        st.subheader("Compensation")
        c = st.columns(4)
        income = c[0].number_input("Monthly income", 1000, 20000, 3000, step=100)
        hike = c[1].number_input("Last salary hike (%)", 11, 25, 13)
        stock = c[2].selectbox("Stock option level", [0, 1, 2, 3])
        daily_rate = c[3].number_input("Daily rate", 100, 1500, int(med.DailyRate))

        st.subheader("Satisfaction")
        c = st.columns(4)
        js = c[0].selectbox("Job satisfaction", [1, 2, 3, 4], index=1,
                            format_func=lambda v: S["JobSatisfaction"][v])
        es = c[1].selectbox("Environment satisfaction", [1, 2, 3, 4], index=1,
                            format_func=lambda v: S["EnvironmentSatisfaction"][v])
        rs = c[2].selectbox("Relationship satisfaction", [1, 2, 3, 4], index=2,
                            format_func=lambda v: S["RelationshipSatisfaction"][v])
        wlb = c[3].selectbox("Work-life balance", [1, 2, 3, 4], index=1,
                             format_func=lambda v: S["WorkLifeBalance"][v])

        st.subheader("Tenure")
        c = st.columns(4)
        years_co = c[0].number_input("Years at company", 0, 40, 2)
        years_role = c[1].number_input("Years in current role", 0, 20, 1)
        years_promo = c[2].number_input("Years since last promotion", 0, 15, 1)
        years_mgr = c[3].number_input("Years with current manager", 0, 20, 1)
        submitted = st.form_submit_button("Predict attrition risk", type="primary")

    if submitted:
        record = {
            "Age": age, "BusinessTravel": travel, "DailyRate": daily_rate, "Department": dept,
            "DistanceFromHome": distance, "Education": education, "EducationField": edu_field,
            "EnvironmentSatisfaction": es, "Gender": gender, "HourlyRate": float(med.HourlyRate),
            "JobInvolvement": involvement, "JobLevel": level, "JobRole": role, "JobSatisfaction": js,
            "MaritalStatus": marital, "MonthlyIncome": income, "MonthlyRate": float(med.MonthlyRate),
            "NumCompaniesWorked": num_companies, "OverTime": overtime, "PercentSalaryHike": hike,
            "PerformanceRating": perf, "RelationshipSatisfaction": rs, "StockOptionLevel": stock,
            "TotalWorkingYears": total_years, "TrainingTimesLastYear": training,
            "WorkLifeBalance": wlb, "YearsAtCompany": years_co, "YearsInCurrentRole": years_role,
            "YearsSinceLastPromotion": years_promo, "YearsWithCurrManager": years_mgr,
        }
        problems = []
        if years_co > total_years:
            problems.append("Years at company cannot be more than total working years.")
        if max(years_role, years_promo, years_mgr) > years_co:
            problems.append("Years in role / since promotion / with manager cannot exceed years at company.")
        if problems:
            for p in problems:
                st.error(p)
            st.stop()

        predictor = load_predictor()
        with st.spinner("Scoring with Spark…"):
            result = predictor.predict(pd.DataFrame([record])).iloc[0]
            drivers = predictor.explain(record)
        p = result.AttritionProbability
        band = result.RiskBand
        g, t = st.columns([1, 1.4])
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=round(100 * p, 1), number={"suffix": "%"},
            title={"text": "Probability of leaving"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": BAND_COLORS[band]},
                   "threshold": {"line": {"color": "black", "width": 3},
                                 "value": 100 * predictor.threshold}}))
        gauge.update_layout(height=280, margin=dict(l=40, r=40, t=60, b=10))
        g.plotly_chart(gauge, use_container_width=True)
        with t:
            st.markdown(f"### Risk: :{'red' if band == 'High' else 'orange' if band == 'Medium' else 'green'}"
                        f"[{band}]")
            st.write(f"Predicted to **{'leave' if result.PredictedAttrition == 'Yes' else 'stay'}** "
                     f"(threshold {predictor.threshold:.2f}). Company-wide attrition is "
                     f"{metrics['dataset_attrition_rate']:.1f}%.")
            st.caption("High = at or above the threshold · Medium = within 40% below it · Low = below that")
        if drivers.empty:
            st.success("No single factor stands out as a risk driver for this employee.")
        else:
            st.markdown("**Top risk drivers and suggested HR actions**")
            st.caption("Impact = how many percentage points the risk drops if this factor were "
                       "at the typical value of employees who stayed.")
            show = drivers.copy()
            for col in ("Current", "Typical (stayed)"):
                show[col] = show[col].map(lambda v: f"{v:g}" if isinstance(v, (int, float)) else v)
            st.dataframe(show, hide_index=True, use_container_width=True,
                         column_config={"Suggested action": st.column_config.TextColumn(width="large")})

# --------------------------------------------------------------------------------------
elif page == "Batch Prediction":
    st.title("Batch prediction")
    st.caption("Upload a CSV of employee records (same columns as the IBM HR dataset) to score "
               "everyone at once with Spark.")
    sample = config.ROOT / "data" / "sample_employees.csv"
    with open(sample, "rb") as f:
        st.download_button("Download a sample input file", f, "sample_employees.csv", "text/csv")
    up = st.file_uploader("Employee CSV", type="csv")
    use_sample = st.checkbox("Or use the sample file", value=up is None)
    if up is not None or use_sample:
        pdf = pd.read_csv(up if up is not None else sample)
        from src.predict import validate
        problems = validate(pdf)
        if problems:
            for p in problems:
                st.error(p)
            st.stop()
        with st.spinner(f"Scoring {len(pdf):,} employees with Spark…"):
            out = load_predictor().predict(pdf)
        counts = out.RiskBand.value_counts()
        k = st.columns(4)
        k[0].metric("Employees scored", f"{len(out):,}")
        for i, b in enumerate(["High", "Medium", "Low"], start=1):
            k[i].metric(f"{b} risk", int(counts.get(b, 0)))
        a, b = st.columns(2)
        pie = px.pie(out, names="RiskBand", color="RiskBand", color_discrete_map=BAND_COLORS,
                     title="Risk bands", hole=0.45)
        a.plotly_chart(pie, use_container_width=True)
        dept = out.groupby("Department")["AttritionProbability"].mean().mul(100).round(1).reset_index()
        b.plotly_chart(px.bar(dept, x="Department", y="AttritionProbability", text="AttritionProbability",
                              title="Average risk by department (%)", color_discrete_sequence=[LEAVE]),
                       use_container_width=True)
        show = ["EmployeeNumber"] if "EmployeeNumber" in out.columns else []
        show += ["AttritionProbability", "RiskBand", "Department", "JobRole", "OverTime",
                 "MonthlyIncome", "YearsAtCompany", "JobSatisfaction"]
        st.dataframe(out.sort_values("AttritionProbability", ascending=False)[show],
                     hide_index=True, use_container_width=True)
        st.download_button("Download predictions", out.to_csv(index=False).encode(),
                           "attrition_predictions.csv", "text/csv")

# --------------------------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model performance")
    st.caption(f"Development / test split: {metrics['split']['development']['rows']} / "
               f"{metrics['split']['test']['rows']} employees (stratified). Each model was tuned with 5-fold "
               "cross-validation on the development set, the threshold was chosen from out-of-fold "
               "predictions, and the model with the best CV AUC was selected. All other numbers are on "
               "the untouched test set.")
    rows = []
    for name, m in metrics["models"].items():
        rows.append({"Model": name, "CV AUC": m["cv_auc"], "OOF AUC": m["oof_auc"],
                     "Test AUC": m["test_auc"], "Accuracy": m["test"]["accuracy"],
                     "Precision": m["test"]["precision"], "Recall": m["test"]["recall"],
                     "F1": m["test"]["f1"], "Threshold": m["test"]["threshold"],
                     "Train time (s)": m["train_seconds"]})
    table = pd.DataFrame(rows).round(3)
    styled = (table.style.format(precision=3).format({"Threshold": "{:.2f}", "Train time (s)": "{:.0f}"})
              .highlight_max(subset=["CV AUC", "Test AUC", "F1"], color="#d6ead3"))
    st.dataframe(styled, hide_index=True, use_container_width=True)
    st.caption(f"Selected: **{metrics['best_model']}** (highest CV AUC). Green = best in column.")
    a, b = st.columns(2)
    roc = metrics["roc_best"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name=metrics["best_model"],
                             line=dict(color=STAY, width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                             line=dict(color="#BBBBBB", dash="dash")))
    fig.update_layout(title=f"ROC curve — {metrics['best_model']}", xaxis_title="False positive rate",
                      yaxis_title="True positive rate", height=400)
    a.plotly_chart(fig, use_container_width=True)
    c = metrics["models"][metrics["best_model"]]["test"]["confusion"]
    cm = px.imshow([[c["tn"], c["fp"]], [c["fn"], c["tp"]]], text_auto=True,
                   x=["Predicted stay", "Predicted leave"], y=["Actually stayed", "Actually left"],
                   color_continuous_scale="Blues", title="Confusion matrix (test set)")
    cm.update_layout(height=400, coloraxis_showscale=False)
    b.plotly_chart(cm, use_container_width=True)
    imp = pd.DataFrame(metrics["feature_importance"][:15])
    imp["value"] = imp.get("signed", imp["importance"])
    imp["effect"] = imp["value"].map(lambda v: "Raises risk" if v > 0 else "Lowers risk")
    fig = px.bar(imp[::-1], x="value", y="feature", orientation="h", color="effect",
                 color_discrete_map={"Raises risk": LEAVE, "Lowers risk": STAY},
                 title="Top 15 features" + (" (coefficients on standardised features)"
                                            if (imp["value"] < 0).any() else " (tree importance)"),
                 labels={"value": "Effect on the odds of leaving", "feature": ""})
    st.plotly_chart(fig.update_layout(height=520), use_container_width=True)

# --------------------------------------------------------------------------------------
else:
    st.title("About this project")
    st.markdown(f"""
**Employee Attrition Prediction using HR Analytics** — Industry domain: HR Analytics.

Employee turnover is expensive: replacing an employee costs a large share of their annual salary
in hiring, onboarding and lost productivity. This app predicts which employees are likely to leave
so HR can act early, and explains *why* for each person.

**How it works**
1. **Data** — IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 attributes).
2. **Processing** — Apache Spark (PySpark DataFrames and Spark SQL): cleaning, constant-column
   removal, five engineered features, stratified 85 / 15 development / test split.
3. **Modelling** — Spark MLlib pipeline (StringIndexer → OneHotEncoder → VectorAssembler →
   StandardScaler → classifier). Logistic Regression, Decision Tree, Random Forest and
   Gradient-Boosted Trees, each tuned with 5-fold cross-validation and class weights for the
   16% minority class.
4. **Selection** — best model by cross-validated ROC-AUC (**{metrics['best_model']}**), decision
   threshold tuned for F1 on out-of-fold predictions; test set used once for the final numbers.
5. **Explanation** — what-if analysis per employee: each factor is reset to the typical value of
   employees who stayed, and the drop in risk shows how much that factor contributes.

**Student:** Varun Mundhada (PRN 23070521184) · B.Tech CSE · Symbiosis Institute of Technology, Nagpur
""")
