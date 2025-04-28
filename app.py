import streamlit as st
import pandas as pd
from food_recommender import evaluate_model, get_recommendations  # Import from the core logic file

# Load data (assuming the CSV is in the same directory)
df = pd.read_csv("food_responses.csv")

# Streamlit inputs for the user
st.title('Food Recommendation System')

food_court = st.selectbox('Select Food Court/Cafe Location', df['Food Court/Cafe Location'].unique())
cuisine = st.selectbox('Select Cuisine Type', df['Cuisine Type'].unique())
veg_nonveg = st.selectbox('Select Veg/Non-veg', df['Veg/Non-veg'].unique())
meal_type = st.selectbox('Select Meal Type', df['Meal Type'].unique())
spice_level = st.selectbox('Select Spice Level', df['Spice Level'].unique())
price = st.slider('Select Price', min_value=int(df['Price'].min()), max_value=int(df['Price'].max()), step=5)
rating = st.slider('Select Minimum Rating', 
                   min_value=float(df['Rating'].min()), 
                   max_value=float(df['Rating'].max()), 
                   step=0.1)

# Get recommendations based on the user input
user_input = {
    'Food Court/Cafe Location': food_court,
    'Cuisine Type': cuisine,
    'Veg/Non-veg': veg_nonveg,
    'Meal Type': meal_type,
    'Spice Level': spice_level,
    'Price': price,
    'Rating': rating
}

# Show recommendations
if st.button('Get Recommendations'):
    recommendations = get_recommendations(user_input)
    st.write("Recommended Food Items:")
    st.dataframe(recommendations)

# Evaluate model and display metrics
r2, rmse, mae = evaluate_model()
st.write(f"Model Evaluation Metrics:")
st.write(f"R²: {r2:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"MAE: {mae:.4f}")
