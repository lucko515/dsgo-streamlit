import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn

st.write("""
# Churn Prediction 
""")

st.sidebar.header('User Input Features')


# Collects user input features into dataframe
# number_customer_service_calls
def user_input_features():

    account_length = st.sidebar.slider('Account Length (in months)', 1, 250, 125)
    international_plan = st.sidebar.selectbox('International plan', ('yes', 'no'))
    voice_mail_plan = st.sidebar.selectbox('Voice Mail Plan', ('yes', 'no'))
    number_vmail_messages = st.sidebar.slider('Number of Voice mail messages', 0, 55, 30)
    total_day_minutes = st.sidebar.slider('Total Day minutes', 0, 352, 200)
    total_day_calls = st.sidebar.slider('Total Day calls', 0, 165, 100)
    total_day_charge = st.sidebar.slider('Total Day charge', 0, 60, 30)
    total_eve_minutes = st.sidebar.slider('Total Evening minutes', 0, 360, 200)
    total_eve_calls = st.sidebar.slider('Total Evening calls', 0, 170, 90)
    total_eve_charge = st.sidebar.slider('Total Evening charge', 0, 31, 15)
    total_night_minutes = st.sidebar.slider('Total Night minutes', 0, 400, 200)
    total_night_calls = st.sidebar.slider('Total Night calls', 0, 180, 90)
    total_night_charge = st.sidebar.slider('Total Night charge', 0, 18, 10)
    total_intl_minutes = st.sidebar.slider('Total International minutes', 0, 20, 10)
    total_intl_calls = st.sidebar.slider('Total International calls', 0, 20, 10)
    total_intl_charge = st.sidebar.slider('Total International charge', 0, 6, 3)
    number_customer_service_calls = st.sidebar.slider('Number of customer service calls', 0, 10, 0)

    data = {'account_length': account_length,
            'international_plan': 1 if international_plan == "yes" else 0,
            'voice_mail_plan': 1 if voice_mail_plan == "yes" else 0,
            'number_vmail_messages': number_vmail_messages,
            'total_day_minutes': total_day_minutes,
            'total_day_calls': total_day_calls,
            'total_day_charge': total_day_charge,
            'total_eve_minutes': total_eve_minutes,
            'total_eve_calls': total_eve_calls,
            'total_eve_charge': total_eve_charge,
            'total_night_minutes': total_night_minutes,
            'total_night_calls': total_night_calls,
            'total_night_charge': total_night_charge,
            'total_intl_minutes': total_intl_minutes,
            'total_intl_calls': total_intl_calls,
            'total_intl_charge': total_intl_charge,
            'number_customer_service_calls': number_customer_service_calls,
            }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()


# Reads in saved saved models
churn_model = joblib.load("models/churn_predictin.joblib")
churn_reason_model = joblib.load("models/reason_prediction.joblib")
scaler = joblib.load("models/scaler.joblib")

scaled_inputs = scaler.transform(input_df.values)


# Apply model to make predictions
prediction = churn_model.predict(scaled_inputs)
st.subheader('Prediction')
st.write(prediction)

# If prediction is 1 (Churn) - show reason predictions
if prediction[0] == 1:
    reason_prediction = churn_reason_model.predict_proba(scaled_inputs)

    st.subheader('Potential reason for churn: ')
    st.write(reason_prediction)
