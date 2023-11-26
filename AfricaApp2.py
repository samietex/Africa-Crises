# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 23:17:17 2023

@author: Windows
"""


import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
model = load('africa_crises_model2.joblib')

def preprocess_categorical(df):
    lb = LabelEncoder()

    df['country'] = lb.fit_transform(df['country'])
    df['banking_crisis'] = lb.fit_transform(df['banking_crisis'])
    return df

def preprocess_numerical(df):
    # Scale numerical columns using StandardScaler
    scaler = MinMaxScaler()
    numerical_cols = ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'inflation_annual_cpi', 'independence', 'currency_crises', 'inflation_crises']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df

def preprocessor(input_df):
    # Preprocess categorical and numerical columns separately
    input_df = preprocess_categorical(input_df)
    input_df = preprocess_numerical(input_df)
    return input_df

def main():
    st.title('Africa Crises Predictor App')
    st.write('This app is built to determine if there is going to be any systemic crises depending on some available features. Please feel free to experiment with the input features below.')

    input_data = {}
    col1, col2 = st.columns(2)

    with col1:
        input_data['country'] = st.selectbox('Country', ['Egypt', 'South Africa', 'Algeria', 'Zimbabwe', 'Angola', 'Morocco', 'Zambia', 'Mauritius', 'Kenya', 'Tunisia', 'Nigeria', 
                                                         'Central African Republic', 'Ivory Coast'])
        input_data['exch_usd'] = st.number_input('Exchange against USD', step=1)
        input_data['domestic_debt_in_default'] = st.number_input('Any Domestic Debt? If yes 1, if No, 0', min_value=0, max_value=1)
        input_data['sovereign_external_debt_default'] = st.number_input('Any Sovereign External Debt? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['inflation_annual_cpi'] = st.number_input('Inflation Annual CPI', step=1)

    with col2:
        input_data['independence'] = st.number_input('Independence? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['currency_crises'] = st.number_input('Currency Crises? if Yes 1, if No, 0', min_value=0, max_value=1)
        input_data['inflation_crises'] = st.number_input('Inflation Crises? if Yes 1', min_value=0, max_value=1)
        input_data['banking_crisis'] = st.selectbox('Banking Crisis?', ['crisis', 'no crisis'])

    input_df = pd.DataFrame([input_data])
    st.write(input_df)

    if st.button('Predict'):
        final_df = preprocessor(input_df)
        prediction = model.predict(final_df)[0]
        
        if prediction == 1:
            st.write('There is a likelihood that there will be systemic crises.')
        else:
            st.write('There is a likelihood that there will not be any systemic crises')
if __name__ == '__main__':
        main()    