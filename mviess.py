import pandas as pd
import pickle 
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

model=pickle.load(open('trained_model_sav', 'rb'))
scaler=pickle.load(open('vectoriser-save', 'rb'))
review=st.text_input('Enter movie review')

if st.button('Predict'):
    #scaler= TfidfVectorizer(max_features=2500)
    review_scale= scaler.transform([review]).toarray()
    result=model.predict(review_scale)
    if result[0]==0:
        st.write('Negative review')
    else:
        st.write('positive review')

