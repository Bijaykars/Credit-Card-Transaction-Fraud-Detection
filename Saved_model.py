import pickle
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
import numpy as np


with open(r"C:\Users\karki\OneDrive\Desktop\GitHub_repo\Credit-Card-Transaction-Fraud-Detection\credit_card_model.pkl", "rb") as file:
    sc, label_encoder, rf_classifier = pickle.load(file)

st.title("Credit Card Fraud Detection Checker")

cc_num = st.number_input("Enter the Credit Card Number",help="The Credit Card Has 16 Digits. Eg: 4929 1010 1234 5678")
merchant = st.text_input("Who is the merchant of the Card")
category = st.text_input("Which Category of Thing did you use it to purchase")
amount = st.number_input("How much Did You Spend?")
zip = st.number_input("What is your Zip Code?")
city_pop = st.number_input("Enter the City population")
job = st.text_input("Enter The Job That you do")

location_name = st.text_input("Enter the location you used the Card From")
geolocator = Nominatim(user_agent="Cred Card Transaction Fraud Detection")
location_info = geolocator.geocode(location_name)

if location_info:
    lat = location_info.latitude
    lon = location_info.longitude
else:
    st.error("Location not found")

merchant_location = st.text_input("Enter the location of the Merchant")
geolocator = Nominatim(user_agent="MyApp")
location_info = geolocator.geocode(merchant_location)

if location_info:
    merch_lat = location_info.latitude
    merch_long = location_info.longitude
else:
    st.error("Merchant location not found")


trans_date_trans_time = st.date_input("Enter the Transaction Date")
dob = st.date_input("Enter Your BirthDay")

data= {"trans_date_trans_time": [trans_date_trans_time],
     "dob": [dob]}

df = pd.DataFrame(data)
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'])
age = pd.DatetimeIndex(df["trans_date_trans_time"]).year - pd.DatetimeIndex(df["dob"]).year
hour = df['trans_date_trans_time'].dt.hour
day = df['trans_date_trans_time'].dt.day
day_of_week = df['trans_date_trans_time'].dt.dayofweek 
month = df['trans_date_trans_time'].dt.month
quarter = df['trans_date_trans_time'].dt.quarter

gender_F = 1
gender_M = 0


merchant_encoded = label_encoder.transform([merchant])[0]
category_encoded = label_encoder.transform([category])[0]
job_encoded = label_encoder.transform([job])[0]
user_input = [cc_num, merchant_encoded, category_encoded, amount, zip, lat, lon, city_pop, job_encoded, merch_lat, merch_long, age[0], hour[0], day[0], day_of_week[0], quarter[0], gender_F, gender_M]
user_input = np.array(user_input).reshape(1, -1)  # Reshape to a 2D array

# Make prediction
prediction = rf_classifier.predict(user_input)

# Display prediction result
st.write("Prediction:", prediction)