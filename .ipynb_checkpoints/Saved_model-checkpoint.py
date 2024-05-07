import pickle
import pandas as pd
import streamlit as st

with open(r"C:\Users\karki\OneDrive\Desktop\GitHub_repo\Credit-Card-Transaction-Fraud-Detection\credit_card_model.pkl", "rb") as file:
    sc, label_encoder, rf_classifier = pickle.load(file)

st.title("Credit Card Fraud Detection Checker")
cc_num = st.number_input("Enter the Credit Card Number")
merchant = st.text_input("Who is the merchant of the Card")
category = st.text_input("Which Category of Thing did you use it to purchase")
amount = st.number_input("How much Did You Spend?")
zip = st.number_input("What is your Zip Code?")
lat = st.number_input("Enter Your Latitude and Longitude")



new_df= pd.read_csv(r"C:\Users\karki\OneDrive\Desktop\GitHub_repo\fraudTest.csv")


new_df1 = new_df[['cc_num', 'merchant', 'category', 'amt', 'zip', 'lat', 'long',
       'city_pop', 'job', 'merch_lat', 'merch_long', 'is_fraud', 'trans_date_trans_time', 'dob',
       'gender']].copy()

new_df1["trans_date_trans_time"] = pd.to_datetime(new_df1['trans_date_trans_time'])
new_df1["age"] = pd.DatetimeIndex(new_df1["trans_date_trans_time"]).year - pd.DatetimeIndex(new_df1["dob"]).year
new_df1['hour'] = new_df1['trans_date_trans_time'].dt.hour
new_df1['day'] = new_df1['trans_date_trans_time'].dt.day
new_df1['day_of_week'] = new_df1['trans_date_trans_time'].dt.dayofweek 
new_df1['month'] = new_df1['trans_date_trans_time'].dt.month
new_df1['quarter'] = new_df1['trans_date_trans_time'].dt.quarter
new_df1.drop(['trans_date_trans_time','dob'],axis=1, inplace= True)
new_df1["merchant"] = new_df1["merchant"].str.removeprefix("fraud_")
new_df1 = pd.get_dummies(new_df1, columns=['gender'], prefix='gender',  dtype= int)
new_df1['job'] = label_encoder.fit_transform(new_df1['job'])
new_df1['merchant'] = label_encoder.fit_transform(new_df1['merchant'])
new_df1['category'] = label_encoder.fit_transform(new_df1['category'])

def preditction(user_input):
    model_output = rf_classifier.predict(user_input)
    
    return user_input



