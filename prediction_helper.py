from joblib import load
import pandas as pd

model_rest = load('artifacts\model_rest.joblib')
model_young = load('artifacts\model_young.joblib')

scaler_rest = load('artifacts\scaler_rest.joblib')
scaler_young = load('artifacts\scaler_young.joblib')

def preprocess_input(selected_values):
    risk_score = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    df = pd.DataFrame([selected_values])
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    if df.at[0, 'gender'] == 'Male':
        df['gender_Male'] = 1
    else:
        df['gender_Male'] = 0

    if df.at[0, 'region'] == 'Northwest':
        df['region_Southeast'] = 0
        df['region_Southwest'] = 0
        df['region_Northwest'] = 1
    elif df.at[0, 'region'] == 'Southeast':
        df['region_Southeast'] = 1
        df['region_Northwest'] = 0
        df['region_Southwest'] = 0
    elif df.at[0, 'region'] == 'Southwest':
        df['region_Southwest'] = 1
        df['region_Northwest'] = 0
        df['region_Southeast'] = 0
    else:
        df['region_Northwest'] = 0
        df['region_Southeast'] = 0
        df['region_Southwest'] = 0

    if df.at[0, 'marital_status'] == 'Unmarried':
        df['marital_status_Unmarried'] = 1
    else:
        df['marital_status_Unmarried'] = 0

    if df.at[0, 'employment_status'] == 'Salaried':
        df['employment_status_Salaried'] = 1
        df['employment_status_Self-Employed'] = 0
    elif df.at[0, 'employment_status'] == 'Self-Employed':
        df['employment_status_Self-Employed'] = 1
        df['employment_status_Salaried'] = 0
    else:
        df['employment_status_Salaried'] = 0
        df['employment_status_Self-Employed'] = 0

    if df.at[0, 'smoking_status'] == 'Occasional':
        df['smoking_status_Occasional'] = 1
        df['smoking_status_Regular'] = 0
    elif df.at[0, 'smoking_status'] == 'Regular':
        df['smoking_status_Regular'] = 1
        df['smoking_status_Occasional'] = 0
    else:
        df['smoking_status_Occasional'] = 0
        df['smoking_status_Regular'] = 0

    if df.at[0, 'bmi_category'] == 'Obesity':
        df['bmi_category_Obesity'] = 1
        df['bmi_category_Overweight'] = 0
        df['bmi_category_Underweight'] = 0
    elif df.at[0, 'bmi_category'] == 'Overweight':
        df['bmi_category_Overweight'] = 1
        df['bmi_category_Obesity'] = 0
        df['bmi_category_Underweight'] = 0
    elif df.at[0, 'bmi_category'] == 'Underweight':
        df['bmi_category_Underweight'] = 1
        df['bmi_category_Obesity'] = 0
        df['bmi_category_Overweight'] = 0
    else:
        df['bmi_category_Obesity'] = 0
        df['bmi_category_Overweight'] = 0
        df['bmi_category_Underweight'] = 0

    df["age"] = selected_values['Age']
    df["number_of_dependants"] = selected_values['Number of Dependants']
    df["income_lakhs"] = selected_values['Income Lakhs']
    plan_map = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df["insurance_plan"] = plan_map[selected_values['Insurance Plan']]
    df["genetical_risk"] = selected_values['Genetical Risk']
    df["medical_history"] = selected_values['Medical History'].lower()
    df["medical_history"] = df["medical_history"] if '&' not in df["medical_history"] else df["medical_history"].split(' & ')
    med_hist = df.loc[0, "medical_history"]
    df['normalized_risk_score'] = 0
    # Medical history risk score
    if '&' in med_hist:
        med_hist_list = [x.strip() for x in med_hist.split('&')]
        df['normalized_risk_score'] = sum([risk_score.get(item, 0) for item in med_hist_list])
    else:
        df['normalized_risk_score'] = risk_score.get(med_hist, 0)

    min_value = df['normalized_risk_score'].min()
    max_value = df['normalized_risk_score'].max()
    print(df.at[0, 'normalized_risk_score'], min_value, max_value)

    if max_value == 0 & min_value == 0:
        df['normalized_risk_score'] = 0
    else:
        df['normalized_risk_score'] = (df.at[0, 'normalized_risk_score'] - min_value) / (max_value - min_value)

    df['income_level'] = 0
    df.drop(['medical_history','gender', 'bmi_category', 'region', 'marital_status', 'employment_status', 'smoking_status'], axis=1, inplace=True)

    if df.at[0, 'age'] <= 25:
        scaler_object = scaler_young
    else:
        scaler_object = scaler_rest

    cols_to_scale = scaler_object['columns']
    scaler = scaler_object['scaler']
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis=1, inplace=True)

    expected_order = ['age', 'number_of_dependants', 'income_lakhs',
'insurance_plan', 'genetical_risk', 'normalized_risk_score', 'gender_Male',
'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
'employment_status_Salaried', 'employment_status_Self-Employed',
'smoking_status_Occasional', 'smoking_status_Regular', 'bmi_category_Obesity',
'bmi_category_Overweight', 'bmi_category_Underweight']

    df = df[expected_order]

    for col in df.columns:
        print(f'{col} value is {df.at[0, col]}')
    return df

def predict(selected_values):
    input_df = preprocess_input(selected_values)
    if selected_values['Age'] <= 25:
        prediction = model_young.predict(input_df)
    else:
        prediction = model_rest.predict(input_df)

    return int(prediction)
