import streamlit as st
import prediction_helper

st.title('Health Insurance Prediction App')

# Define your features and options in a list of dicts
features = [
    {"label": "Gender", "options": ['Male', 'Female'], "type": "select"},
    {"label": "Region", "options": ['Northwest', 'Southeast', 'Northeast', 'Southwest'], "type": "select"},
    {"label": "Marital Status", "options": ['Unmarried', 'Married'], "type": "select"},
    {"label": "BMI Category", "options": ['Normal', 'Obesity', 'Overweight', 'Underweight'], "type": "select"},
    {"label": "Smoking Status", "options": ['No Smoking', 'Regular', 'Occasional'], "type": "select"},
    {"label": "Employment Status", "options": ['Salaried', 'Self-Employed', 'Freelancer'], "type": "select"},
    {"label": "Medical History", "options": [
        'Diabetes', 'High blood pressure', 'No Disease',
        'Diabetes & High blood pressure', 'Thyroid', 'Heart disease',
        'High blood pressure & Heart disease', 'Diabetes & Thyroid',
        'Diabetes & Heart disease'
    ], "type": "select"},
    {"label": "Insurance Plan", "options": ['Bronze', 'Silver', 'Gold'], "type": "select"}
]

# Configuration for numerical features
num_features = [
    {"label": "Age", "type": "number", "min": 0, "max": 100},
    {"label": "Number of Dependants", "type": "number", "min": 0, "max": 5},
    {"label": "Income Lakhs", "type": "number", "min": 0, "max": 100},
    {"label": "Genetical Risk", "type": "number", "min": 0, "max": 5}
]

# Combine all features
all_features = features + num_features

# Display features in rows of 3
selected_values = {}
for i in range(0, len(all_features), 3):
    cols = st.columns(3)
    for j, feature in enumerate(all_features[i:i+3]):
        with cols[j]:
            if feature.get("type") == "select":
                selected_values[feature["label"]] = st.selectbox(feature["label"], feature["options"])
            elif feature.get("type") == "number":
                selected_values[feature["label"]] = st.number_input(
                    feature["label"],
                    min_value=feature["min"],
                    max_value=feature["max"]
                )


if st.button("Predict"):
    print("Health Insurance Prediction App button is pressed")
    prediction = prediction_helper.predict(selected_values)
    st.success(f"Predicted Health Insurance Cost is: {prediction}")

