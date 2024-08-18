import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib
import json

#reading Data
data=pd.read_csv("HistoricalQuotes.csv")
#cleaning data
data=data.dropna()

data[' Open']=data[' Open'].str.replace('$','')
data[' High']=data[' High'].str.replace('$','')
data[' Low']=data[' Low'].str.replace('$','')
data[' Close/Last']=data[' Close/Last'].str.replace('$','')
x=data.drop(columns=['Date',' Close/Last'])
y=data[' Close/Last']
#splitting data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
#training model
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
#saving model
joblib.dump(rf,'model.pkl')
#Accuracy Testing
y_predict=rf.predict(x_test)
r2=r2_score(y_test,y_predict)
metrice={
     'accuracy':r2
}
with open('metrice.json','w') as f:
     json.dump(metrice,f,indent=1)





