import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import base64


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


def car_sort(target_company):
    filtered_df = car_info[car_info['company'] == target_company]
    car_names = filtered_df['name'].tolist()
    car_names = pd.Series(car_names).unique()
    return filtered_df, car_names


def year_sort(filtered_df, target_car):
    filtered_df = filtered_df[filtered_df['name'] == target_car]
    car_years = filtered_df['year'].tolist()
    car_years = pd.Series(car_years).unique()
    return filtered_df, car_years

def fuel_sort(filtered_df, target_car):
    car_fuel = filtered_df['fuel_type'].tolist()
    car_fuel = pd.Series(car_fuel).unique()
    return filtered_df, car_fuel


car_csv = pd.read_csv('cleaned_car.csv')
car_info = pd.DataFrame(car_csv)
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))

add_bg_from_local('bgimage.jpeg')

st.title('Car Price Predictor')

company = st.selectbox(
    'Select Car Company:',
    np.unique(car_info['company'].values)
)


filtered_df, car_models = car_sort(company)

car_model = st.selectbox(
    'Select Car Model:',
    car_models
)

filtered_df, car_years = year_sort(filtered_df, car_model)

year = st.selectbox(
    'Select the Year of Manufacturing:',
    np.sort(car_years)
)

filtered_df, car_fuel = fuel_sort(filtered_df, car_model)

fuel = st.selectbox(
    'Select the fuel type:',
    car_fuel
)

kms = st.text_input(
    'Enter the KMs Travelled:'
)

test = (car_model, company, year, kms, fuel)

if st.button('Predict'):
    prediction = np.round(model.predict(pd.DataFrame([[car_model, company, year, kms, fuel]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))[0],2)
    st.subheader("Your " + car_model + " will cost you around:")
    st.subheader("\u20B9" + str(prediction))

