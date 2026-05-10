import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="<---repo id---->/tourism_package_model", filename="best_tourism_package_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism App Package Prediction")
st.write("""
This application predicts whether a customer is likely to purchase the Wellness Tourism Package.
Based on customer demographics, travel behavior, and interaction details."
""")

# User input
age = st.slider("Age", min_value=18, max_value=70, value=30)

typeof_contact = st.selectbox(
    "Type of Contact",
    ["Company Invited", "Self Inquiry"]
)

city_tier = st.selectbox(
    "City Tier",
    [1, 2, 3]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Freelancer", "Large Business"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

number_of_person_visiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    max_value=10,
    value=2
)

preferred_property_star = st.selectbox(
    "Preferred Property Star",
    [1, 2, 3, 4, 5]
)

marital_status = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

number_of_trips = st.number_input(
    "Number of Trips per Year",
    min_value=0,
    max_value=20,
    value=2
)

passport = st.selectbox(
    "Passport Availability",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

number_of_children_visiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    max_value=5,
    value=0
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=1000,
    max_value=500000,
    value=30000,
    step=1000
)

# Customer Interaction Data
pitch_satisfaction_score = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

product_pitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

number_of_followups = st.number_input(
    "Number of Follow-Ups",
    min_value=0,
    max_value=10,
    value=2
)

duration_of_pitch = st.number_input(
    "Duration of Pitch (Minutes)",
    min_value=5,
    max_value=60,
    value=20
)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'ProductPitched': product_pitched,
    'NumberOfFollowups': number_of_followups,
    'DurationOfPitch': duration_of_pitch
}])

# Predict button
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.error("Customer is unlikely to purchase the Wellness Tourism Package.")
