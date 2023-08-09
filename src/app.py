import streamlit as st
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import re
import numpy as np
from features.build_features import FeatureEngineering,DataProcessing, get_class




path='C:/Users/dvice/Desktop/daata/orders.csv'
#df=pd.read_csv(path)

# Load the trained model and data transformation pipeline
with open('C:/Users/dvice/Desktop/daata/Rappi_Project/rappi_challenge/models/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


st.title('Rappi Demo')

st.write("Upload a CSV file:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(path)
    data_processing = DataProcessing()
    processed_data = data_processing.process_data(df)
    prediction = model.predict(processed_data.values)
    value=get_class(prediction)
    st.write("Predictions:")
    st.write(prediction)
