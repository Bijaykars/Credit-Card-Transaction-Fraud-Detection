import pickle
import pandas as pd

with open(r"C:\Users\karki\OneDrive\Desktop\GitHub_repo\Credit-Card-Transaction-Fraud-Detection\credit_card_model.pkl", "rb") as file:
    sc, label_encoder, rf_classifier = pickle.load(file)

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

X_train = new_df1.drop('is_fraud', axis= 1).to_numpy()

y_train = new_df1["is_fraud"].to_numpy()

X_train = sc.fit_transform(X_train)

rf_classifier.predict(X_train)