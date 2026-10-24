
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------
# 1. Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Student Performance Prediction")
st.write(
    "Predict student marks using study hours, "
    "attendance, and previous marks."
)

st.divider()

# --------------------------------
# 2. Upload Dataset
# --------------------------------
st.header("1. Dataset Upload")

uploaded_file = st.file_uploader(
    "Upload student CSV file",
    type=["csv"],
    key="student_csv_upload"
)

try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("student_data.csv")

except FileNotFoundError:
    st.error(
        "student_data.csv not found. "
        "Upload a CSV file to continue."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not read CSV: {error}")
    st.stop()

required_columns = [
    "study_hours",
    "attendance",
    "previous_marks",
    "marks"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing CSV columns: {missing_columns}")
    st.stop()

df = df[required_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

df = df.dropna()

if len(df) < 10:
    st.error(
        "Please provide at least 10 valid student records "
        "for model evaluation."
    )
    st.stop()

if (
    (df["study_hours"] < 0).any()
    or (df["study_hours"] > 24).any()
    or (df["attendance"] < 0).any()
    or (df["attendance"] > 100).any()
    or (df["previous_marks"] < 0).any()
    or (df["previous_marks"] > 100).any()
    or (df["marks"] < 0).any()
    or (df["marks"] > 100).any()
):
    st.error(
        "Invalid data found. Hours must be 0–24 "
        "and attendance/marks must be 0–100."
    )
    st.stop()

st.success(f"Dataset loaded: {len(df)} valid records")

with st.expander("View Dataset"):
    st.dataframe(df, use_container_width=True)

# --------------------------------
# 3. Dataset Overview
# --------------------------------
st.header("2. Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))
col2.metric("Input Features", 3)
col3.metric("Target", "Marks")

# --------------------------------
# 4. Prepare Data
# --------------------------------
X = df[
    ["study_hours", "attendance", "previous_marks"]
]

y = df["marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------
# 5. Train Models and Evaluate
# --------------------------------
st.header("3. Model Training and Comparison")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
}

trained_models = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    # R² requires at least two test samples
    if len(y_test) >= 2:
        r2 = r2_score(y_test, test_predictions)
    else:
        r2 = float("nan")

    trained_models[name] = model

    results.append({
        "Model": name,
        "MAE": mae,
        "R2 Score": r2
    })

results_df = pd.DataFrame(results)

st.dataframe(
    results_df.round(3),
    use_container_width=True
)

st.caption(
    "Lower MAE is generally better. Higher R² is generally "
    "better, but R² can be negative. Results depend on the dataset."
)

# --------------------------------
# 6. Realtime Prediction
# --------------------------------
st.header("4. Live Student Marks Prediction")

st.write(
    "Change any input below to calculate an updated estimate."
)

selected_model_name = st.selectbox(
    "Select Machine Learning Model",
    list(trained_models.keys()),
    key="live_model_selection"
)

selected_model = trained_models[selected_model_name]

col1, col2, col3 = st.columns(3)

with col1:
    hours = st.number_input(
        "Daily Study Hours",
        min_value=0.0,
        max_value=24.0,
        value=5.0,
        step=0.5,
        key="live_hours"
    )

with col2:
    attendance = st.number_input(
        "Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0,
        key="live_attendance"
    )

with col3:
    previous_marks = st.number_input(
        "Previous Marks",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0,
        key="live_previous_marks"
    )

new_student = pd.DataFrame({
    "study_hours": [hours],
    "attendance": [attendance],
    "previous_marks": [previous_marks]
})

prediction = selected_model.predict(new_student)[0]
prediction = max(0, min(100, float(prediction)))

st.divider()
st.subheader("Predicted Result")

st.metric(
    "Estimated Student Marks",
    f"{prediction:.2f} / 100"
)

st.caption(
    "This is an estimate from a trained model, not a guarantee "
    "of the student's actual examination result."
)

# --------------------------------
# 7. Download Prediction as CSV
# --------------------------------
result_df = pd.DataFrame([{
    "study_hours": hours,
    "attendance": attendance,
    "previous_marks": previous_marks,
    "predicted_marks": round(prediction, 2),
    "model": selected_model_name
}])

csv_data = result_df.to_csv(
    index=False
).encode("utf-8-sig")

st.download_button(
    label="Download Prediction Results (CSV)",
    data=csv_data,
    file_name="prediction_results.csv",
    mime="text/csv",
    key="download_prediction_csv"
)

# --------------------------------
# 8. Graph
# --------------------------------
st.header("5. Study Hours vs Student Marks")

fig, ax = plt.subplots(figsize=(9, 4))

ax.scatter(
    df["study_hours"],
    df["marks"],
    label="Dataset"
)

ax.scatter(
    [hours],
    [prediction],
    marker="x",
    s=150,
    label="Predicted Student"
)

ax.set_xlabel("Daily Study Hours")
ax.set_ylabel("Marks")
ax.set_title("Study Hours vs Student Marks")
ax.grid(True)
ax.legend()

st.pyplot(fig)
plt.close(fig)

# --------------------------------
# 9. Project Information
# --------------------------------
st.divider()
st.subheader("About This Project")

st.write("""
**Project:** Student Performance Prediction

**Technology:** Python, Streamlit, Pandas,
Matplotlib, Scikit-learn

**Models:** Linear Regression and Random Forest

**Inputs:** Study Hours, Attendance, Previous Marks

**Output:** Estimated Student Marks
""")

st.warning(
    "Use sufficient, accurate, and appropriately collected "
    "student data. Synthetic demonstration data cannot establish "
    "real-world prediction accuracy."
)
