import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ---------------------------------------------------------
# Diabetes Prediction - Streamlit GUI
# Model/features follow Diabetes_Prediction.ipynb
# ---------------------------------------------------------

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Diabetes Prediction")
st.write("Enter the details below to run the machine-learning model.")

st.info(
    "Educational project demo only. This result is not a medical diagnosis "
    "and must not be used to make healthcare decisions."
)

MODEL_PATH = Path(__file__).with_name("diabetes_model.pkl")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

if model is None:
    st.error(
        "Model file not found. Keep 'diabetes_model.pkl' in the same folder "
        "as this Streamlit file, then run the app again."
    )
    st.stop()

st.subheader("Patient Information")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=45, step=1)
        gender_label = st.selectbox("Gender", ["Female", "Male", "Unknown"])
        bmi = st.number_input("BMI", min_value=1.0, max_value=100.0, value=28.5, step=0.1)
        glucose = st.number_input("Glucose", min_value=1.0, max_value=1000.0, value=120.0, step=1.0)
        blood_pressure = st.number_input(
            "Blood Pressure", min_value=1, max_value=300, value=125, step=1
        )
        cholesterol = st.number_input(
            "Cholesterol", min_value=1.0, max_value=1000.0, value=210.0, step=1.0
        )

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=5000.0, value=110.0, step=1.0)
        hba1c = st.number_input("HbA1c", min_value=0.0, max_value=30.0, value=6.0, step=0.1)
        smoking_label = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
        activity_label = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
        family_label = st.selectbox("Family History of Diabetes", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # These mappings match the encoding used in the notebook.
    gender_map = {"Female": 0, "Male": 1, "Unknown": 2}
    smoking_map = {"Never": 0, "Former": 1, "Current": 2}
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    family_map = {"No": 0, "Yes": 1}

    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender_map[gender_label],
        "BMI": bmi,
        "Glucose": glucose,
        "Blood_Pressure": blood_pressure,
        "Cholesterol": cholesterol,
        "Insulin": insulin,
        "HbA1c": hba1c,
        "Smoking_Status": smoking_map[smoking_label],
        "Physical_Activity": activity_map[activity_label],
        "Family_History": family_map[family_label],
    }])

    expected_features = [
        "Age", "Gender", "BMI", "Glucose", "Blood_Pressure",
        "Cholesterol", "Insulin", "HbA1c", "Smoking_Status",
        "Physical_Activity", "Family_History"
    ]

    try:
        # Preserve the exact feature order used to train the notebook model.
        input_data = input_data[expected_features]
        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")
        if prediction == 1:
            st.warning("The model prediction is: Diabetes")
        else:
            st.success("The model prediction is: No Diabetes")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write("Model probability estimates:")
            probability_table = pd.DataFrame({
                "Class": ["No Diabetes", "Diabetes"],
                "Estimated probability": [
                    f"{probabilities[0] * 100:.2f}%",
                    f"{probabilities[1] * 100:.2f}%"
                ]
            })
            st.dataframe(probability_table, hide_index=True, use_container_width=True)

        st.caption(
            "A machine-learning prediction can be wrong. Please consult a "
            "qualified healthcare professional for interpretation or testing."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.write(
            "Check that this model file was created from the same notebook "
            "features and compatible Python/scikit-learn versions."
        )

st.divider()
st.caption("Project model: GradientBoostingClassifier saved as diabetes_model.pkl")
