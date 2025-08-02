import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

df=pd.read_csv("spam mail.csv")
print(df.head(2))
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
print(df.isnull().sum())
df['Category']=df['Category'].replace(['ham','Spam'],['Not spam','Spam'])
print(df.head())
mess=df['Messages']
cat=df['Category']
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)
cv=CountVectorizer(stop_words='english')
cv.fit_transform(mess_train)
features=cv.fit_transform(mess_train)
#Creating model
model=MultinomialNB()
model.fit(features,cat_train)
#test our model
features_test=cv.transform(mess_test) 
# print(model.score(features_test,cat_test))

#predict Data
def predict(message):
  input_message=cv.transform([message]).toarray()
  result=model.predict(input_message)
  return result
st.header('Spam Detection')

input_mess=st.text_input('Enter Message Here')

if st.button('Validate'):
  output=predict(input_mess)
  st.markdown(output)
  
