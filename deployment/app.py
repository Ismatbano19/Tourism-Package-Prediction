import os
import streamlit as st
import pandas as pd
import joblib

# Load the trained tourism prediction model
model_path = os.path.join(
    os.path.dirname(__file__), "best_tourism.joblib"
)
model = joblib.load(model_path)

st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Tourism Package Prediction")

st.write(
    "Predict whether a customer is likely to purchase the "
    "Wellness Tourism Package using customer and interaction data."
)

st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    contact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Inquiry"]
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced"]
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0, value=25000, step=1000
    )
    preferred_star = st.selectbox(
        "Preferred Property Star", [3, 4, 5], index=1
    )

with col2:
    persons_visiting = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=20, value=2
    )
    children_visiting = st.number_input(
        "Children Below Age 5", min_value=0, max_value=10, value=0
    )
    number_of_trips = st.number_input(
        "Number of Trips Annually", min_value=0, max_value=50, value=3
    )
    passport = st.selectbox(
        "Has Passport?", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    own_car = st.selectbox(
        "Owns a Car?", [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )
    pitch_score = st.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5], index=3
    )

st.subheader("Sales Interaction Details")

col3, col4 = st.columns(2)

with col3:
    duration = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0.0, max_value=180.0, value=15.0, step=1.0
    )
    followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=30, value=3
    )

with col4:
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    designation = st.selectbox(
        "Customer Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )

# Build input dataframe using the dataset's feature names
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": persons_visiting,
    "PreferredPropertyStar": preferred_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": passport,
    "OwnCar": own_car,
    "NumberOfChildrenVisiting": children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": followups,
    "DurationOfPitch": duration
}])

if st.button("Predict Purchase", type="primary"):
    try:
        # Match the exact feature order used during model training
        if hasattr(model, "feature_names_in_"):
            input_data = input_data.reindex(
                columns=model.feature_names_in_
            )

        prediction = int(model.predict(input_data)[0])

        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(input_data)[0][1])
            st.metric(
                "Estimated Purchase Probability",
                f"{probability:.1%}"
            )

        if prediction == 1:
            st.success(
                "The model predicts that the customer is likely "
                "to purchase the tourism package."
            )
        else:
            st.info(
                "The model predicts that the customer is less likely "
                "to purchase the tourism package."
            )

        st.caption(
            "This prediction is an estimate based on historical data "
            "and should support, not replace, marketing decisions."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the deployed model was trained with the same "
            "feature names and preprocessing pipeline."
        )
