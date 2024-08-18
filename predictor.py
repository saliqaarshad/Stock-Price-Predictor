import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import json

rf=joblib.load('model.pkl')
st.write('Our Model Predicts the Price of the Stock at the end of the day')
st.write('Enter the following values of your stock')
Volume=st.number_input('Enter Volumn of your Stock')
Open=st.number_input('Enter the Opening price of your Stock')
Low=st.number_input('Enter the Lowest price of your Stock')
High=st.number_input('Enter the Highest price of your Stock')
Data={
     ' Volume':[Volume],' Open':[Open],' High':[High],' Low':[Low]
}
input=pd.DataFrame(Data)
result=rf.predict(input)
if st.button('button'):
     st.write(f"Estimated Closing price of your stock is: {result[0]:.3f}")
with open('metrice.json', 'r') as f:
     metrice = json.load(f)

st.write(f" accuracy: {metrice['accuracy']:.4f}")





