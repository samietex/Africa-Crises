# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 20:03:43 2023

@author: Windows
"""

import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import MinMaxScaler

model = load('africa_crises_model2.joblib')

def cleaner(df):
    cleaned_df = df.copy()
    # Dropping columns with more than 30% missing values
    cols_to_drop = [col for col in cleaned_df.columns if (cleaned_df[col].isnull().sum() / len(cleaned_df)) * 100 > 30]
    cleaned_df.drop(columns=cols_to_drop, inplace=True)
    
    # Filling missing values for object columns with mode and numerical columns with mean
    for col in cleaned_df.columns:
        if cleaned_df[col].dtype == 'object':
            cleaned_df[col].fillna(cleaned_df[col].mode()[0], inplace=True)
        else:
            cleaned_df[col].fillna(cleaned_df[col].mean(), inplace=True)
    
    return cleaned_df

def preprocessor(df):
    cleaned_df = df.copy()
    mm = MinMaxScaler()
    
    # Handling object columns by one-hot encoding
    cleaned_df = pd.get_dummies(cleaned_df, columns = ['country','banking_crisis']) 
    
    # Scaling numerical columns using MinMaxScaler
    numerical_columns = cleaned_df.select_dtypes(exclude=['object']).columns
    cleaned_df[numerical_columns] = mm.fit_transform(cleaned_df[numerical_columns])
    
    # Handling outliers based on specific columns
    '''min_threshold = cleaned_df['inflation_annual_cpi'].quantile(0.05)
    max_threshold = cleaned_df['inflation_annual_cpi'].quantile(0.95)
    min_threshold1 = cleaned_df['exch_usd'].quantile(0.05)
    max_threshold1 = cleaned_df['exch_usd'].quantile(0.95)
    
    cleaned_df = cleaned_df[
        (cleaned_df['inflation_annual_cpi'] > min_threshold) &
        (cleaned_df['inflation_annual_cpi'] < max_threshold) &
        (cleaned_df['exch_usd'] > min_threshold1) &
        (cleaned_df['exch_usd'] < max_threshold1)
    ]'''
    
    return cleaned_df



    def main():
        st.title('Africa Crises Predictor App')
        st.write('This app is built to determine if there is going to be any systemic crises depending on some available features. Please feel free to experiment with the input features below.')
        
        
        input_data = {}
        col1, col2 = st.columns(2)
        
        with col1:
            input_data['country'] = st.selectbox('Country', ['Egypt', 'South Africa', 'Algeria', 'Zimbabwe', 'Angola', 'Morocco', 'Zambia', 'Mauritius', 'Kenya', 'Tunisia', 'Nigeria', 
                                                             'Central African Republic', 'Ivory Coast'])
            input_data['exch_usd'] = st.number_input('Exchange against USD', step = 1)
            input_data['domestic_debt_in_default'] = st.number_input('Any Domestic Debt? If yes 1, if No, 0', min_value = 0, max_value = 1)
            input_data['sovereign_external_debt_default'] = st.number_input('Any Sovereign External Debt? if Yes 1, if No, 0', min_value = 0, max_value = 1)
            input_data['inflation_annual_cpi'] = st.number_input('Inflation Annual CPI', step = 1)
        
        with col2:
            input_data['independence'] = st.number_input('Independence? if Yes 1, if No, 0', min_value = 0, max_value = 1)
            input_data['currency_crises'] = st.number_input('Currency Crises? if Yes 1, if No, 0', min_value = 0, max_value = 1)
            input_data['inflation_crises'] = st.number_input('Inflation Crises? if Yes 1', min_value = 0, max_value = 1)
            input_data['banking_crisis'] = st.selectbox('Banking Crisis?', ['crisis', 'no crisis'])
            
        input_df = pd.DataFrame([input_data])
        st.write(input_df)
        if st.button('Predict'):
            
            #input_df = cleaner(input_df)
            final_df = preprocessor(input_df)
            print(final_df.shape)
            
            prediction = model.predict(final_df)[0]
            print(final_df.head())
            
            if prediction == 1:
                st.write('There is a likelihood that there will be systemic crises.')
            else:
                st.write('There is a likelihood that there will not be any systemic crises')
    if __name__ == '__main__':
        main()    
    
    
    
    
    
    
    
                
                
            
    