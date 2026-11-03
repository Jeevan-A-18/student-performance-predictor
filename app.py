# app.py
# ------------------------------------------------------------------
# This file builds the Streamlit web application.
# It imports ML functions from model.py and presents them
# through a clean, beginner-friendly interface.
# ------------------------------------------------------------------

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Import our ML helper functions from model.py
from model import (
    load_dataset,
    prepare_data,
    split_data,
    train_model,
    evaluate_model,
    predict_marks,
)

# ── Page configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📚",
    layout="centered",
)

# ── Load data & train model (cached so it only runs once) ─────────
@st.cache_resource
def setup_model():
    """Load dataset, train the model, and return everything needed by the UI."""
    df            = load_dataset("dataset.csv")
    X, y          = prepare_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model         = train_model(X_train, y_train)
    mae, r2, _    = evaluate_model(model, X_test, y_test)
    return model, df, X_train, X_test, y_train, y_test, mae, r2

model, df, X_train, X_test, y_train, y_test, mae, r2 = setup_model()


# ════════════════════════════════════════════════════════════════
# SECTION A – Home / Header
# ════════════════════════════════════════════════════════════════
st.title("📚 Student Performance Predictor")
st.markdown(
    """
    Welcome! This application uses **Linear Regression** — one of the simplest
    Machine Learning algorithms — to predict a student's expected examination marks
    based on the number of hours they study.

    > **How it works:** We trained a model on a dataset of students' study hours
    > and their corresponding marks. The model learned the relationship between
    > them and can now predict marks for any number of study hours you provide.
    """
)

st.divider()


# ════════════════════════════════════════════════════════════════
# SECTION B – Prediction
# ════════════════════════════════════════════════════════════════
st.header("🎯 Predict Your Marks")
st.markdown("Use the slider below to enter the number of hours you plan to study:")

# Slider: user picks study hours between 1 and 12
study_hours = st.slider(
    label="Study Hours",
    min_value=1.0,
    max_value=12.0,
    value=6.0,          # default value shown on first load
    step=0.5,
    help="Drag the slider to set your daily study hours",
)

# Make a prediction using the trained model
predicted = predict_marks(model, study_hours)

# Display result in a highlighted metric box
st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="⏱️ Study Hours", value=f"{study_hours} hrs")
with col2:
    st.metric(label="📝 Predicted Marks", value=f"{predicted} / 100")

# Motivational message based on predicted marks
if predicted >= 85:
    st.success("🌟 Excellent! You're on track for outstanding results. Keep it up!")
elif predicted >= 60:
    st.info("👍 Good effort! A little more study could push you even higher.")
elif predicted >= 40:
    st.warning("📖 You're passing, but consider increasing your study hours.")
else:
    st.error("⚠️ Low predicted marks. Try to dedicate more time to studying.")

st.divider()


# ════════════════════════════════════════════════════════════════
# SECTION C – Dataset & Visualization
# ════════════════════════════════════════════════════════════════
st.header("📊 Dataset & Visualization")

# --- 1. Raw data table ---
with st.expander("📋 View Raw Dataset", expanded=False):
    st.dataframe(
        df.rename(columns={"study_hours": "Study Hours", "marks": "Marks"}),
        use_container_width=True,
    )
    st.caption(f"Total records: {len(df)}")

# --- 2. Scatter plot + regression line ---
st.subheader("Scatter Plot with Regression Line")
st.markdown(
    "Each **blue dot** represents a student in our dataset. "
    "The **red line** is what the Linear Regression model learned — "
    "it shows the general trend between study hours and marks."
)

fig, ax = plt.subplots(figsize=(8, 5))

# Plot the actual data points
ax.scatter(
    df["study_hours"],
    df["marks"],
    color="#4C72B0",
    edgecolors="white",
    s=80,
    label="Actual Data",
    zorder=3,
)

# Draw the regression line
x_line = np.linspace(df["study_hours"].min(), df["study_hours"].max(), 200)
y_line = model.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, color="#DD4949", linewidth=2.5, label="Regression Line")

# Highlight the user's prediction point
ax.scatter(
    [study_hours],
    [predicted],
    color="#2CA02C",
    s=150,
    zorder=5,
    marker="*",
    label=f"Your Prediction ({study_hours} hrs → {predicted} marks)",
)

ax.set_xlabel("Study Hours", fontsize=12)
ax.set_ylabel("Examination Marks", fontsize=12)
ax.set_title("Study Hours vs Examination Marks", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, linestyle="--", alpha=0.4)

st.pyplot(fig)
plt.close(fig)   # free memory

st.divider()


# ════════════════════════════════════════════════════════════════
# SECTION D – Model Evaluation
# ════════════════════════════════════════════════════════════════
st.header("🔍 Model Evaluation")
st.markdown(
    "After training, we test the model on data it has **never seen before** "
    "to measure how accurately it generalises to new students."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📉 MAE",
        value=f"{mae:.2f}",
        help="Mean Absolute Error — average difference between predicted and actual marks (lower is better)",
    )
with col2:
    st.metric(
        label="📈 R² Score",
        value=f"{r2:.4f}",
        help="How well the model fits the data. 1.0 = perfect fit.",
    )
with col3:
    st.metric(
        label="🏋️ Training Size",
        value=f"{len(X_train)} records",
        help="Number of student records used to train the model",
    )
with col4:
    st.metric(
        label="🧪 Testing Size",
        value=f"{len(X_test)} records",
        help="Number of student records used to evaluate the model",
    )

# Plain-English explanation of the metrics
st.markdown("---")
st.markdown(
    f"""
    **What do these numbers mean?**

    - **MAE = {mae:.2f}** → On average, the model's prediction is off by about
      **{mae:.1f} marks**. That is quite accurate for a simple model!
    - **R² = {r2:.4f}** → The model explains **{r2 * 100:.1f}%** of the variation
      in student marks just from study hours alone. A value close to **1.0** is great.
    - We used **{len(X_train)} records** to train and **{len(X_test)} records** to test,
      which is an **80 / 20 split** — a common practice in Machine Learning.
    """
)

st.divider()

# ── Footer ────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:grey; font-size:13px;'>"
    "Built with ❤️ using Python · Scikit-learn · Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
