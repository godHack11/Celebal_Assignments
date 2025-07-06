Deploying Machine Learning Models with Streamlit
Develop a web application using Streamlit to deploy a trained machine learning model. The app should allow users to input data, receive predictions, and understand model outputs through 
visualizations. This task will help you learn how to make your models accessible and interactive.


#code

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('50_Startups.csv')
        return df
    except FileNotFoundError:
        st.error("CSV file not found. Make sure '50_Startups.csv' is in the same folder as this script.")
        return pd.DataFrame()

def train_model(df):
    if 'State' in df.columns:
        le = LabelEncoder()
        df['State'] = le.fit_transform(df['State'])  

    X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
    y = df['Profit']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, df
#App
def main():
    st.set_page_config(page_title="Startup Profit Predictor", layout="centered")
    st.title('Startup Profit Prediction App')
    st.write('Enter your startup expenditure to predict the estimated profit.')

    df = load_data()

    if df.empty:
        st.stop()

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    model, df = train_model(df)

    st.subheader("Enter Your Startup's Spending")

    rd_spend = st.number_input('R&D Spend ($)', min_value=0.0, value=50000.0)
    admin_spend = st.number_input('Administration Spend ($)', min_value=0.0, value=50000.0)
    marketing_spend = st.number_input('Marketing Spend ($)', min_value=0.0, value=50000.0)

    if st.button('Predict Profit'):
        input_data = np.array([[rd_spend, admin_spend, marketing_spend]])
        predicted_profit = model.predict(input_data)[0]
        st.success(f'💰 Estimated Profit: ${predicted_profit:,.2f}')

        fig = px.scatter_3d(df, x='R&D Spend', y='Marketing Spend', z='Profit',
                            title='Startup Spend vs Profit (3D)',
                            opacity=0.7)
        fig.add_scatter3d(x=[rd_spend], y=[marketing_spend], z=[predicted_profit],
                          mode='markers',
                          marker=dict(size=10, color='red'),
                          name='Your Prediction')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
    
    
# To tun the app, save this script as 'streamlit run startup_profit_app.py' and run it in your terminal.      
