import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load and prepare dataset
df = pd.read_csv("food_responses.csv")
df = df.drop(columns=["Timestamp"])

# Define categorical features and numerical features
categorical_features = ['Food Court/Cafe Location', 'Cuisine Type', 'Veg/Non-veg', 'Meal Type', 'Spice Level']
numerical_features = ['Price']

# One-hot encode the categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_features])

# Create a DataFrame from the encoded categorical variables
encoded_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))

# Combine the encoded categorical data with the numerical features
X = pd.concat([encoded_df, df[numerical_features]], axis=1)
y = df['Rating']

# Train RandomForest Regressor model
rating_model = RandomForestRegressor(n_estimators=100, random_state=42)
rating_model.fit(X, y)

# Predict ratings
df['Predicted_Rating'] = rating_model.predict(X)

# Model Evaluation
r2 = r2_score(y, df['Predicted_Rating'])
rmse = np.sqrt(mean_squared_error(y, df['Predicted_Rating']))
mae = mean_absolute_error(y, df['Predicted_Rating'])

def evaluate_model():
    return r2, rmse, mae

def get_recommendations(user_input):
    # Encode the user input using the same encoder
    user_input_array = encoder.transform([[user_input['Food Court/Cafe Location'],
                                           user_input['Cuisine Type'],
                                           user_input['Veg/Non-veg'],
                                           user_input['Meal Type'],
                                           user_input['Spice Level']]])

    # Convert the user input to a DataFrame
    user_vector = pd.DataFrame(user_input_array, columns=encoder.get_feature_names_out(categorical_features))

    # Add numerical features to the user_vector
    user_vector["Price"] = user_input["Price"]
    user_vector["Rating"] = user_input["Rating"]

    # Get all columns from the encoded dataframe
    all_columns = list(encoded_df.columns) + ['Price', 'Rating']

    # Ensure user_vector has the same columns as the dataset (encoded features + Price + Rating)
    user_vector = user_vector.reindex(columns=all_columns, fill_value=0)  # Fill missing columns with 0

    # Now `user_vector` and `all_features` should have the same columns
    all_features = pd.concat([encoded_df, df[numerical_features]], axis=1)

    # Ensure all columns in all_features are present in user_vector
    all_features = all_features.reindex(columns=all_columns, fill_value=0)

    # Compute cosine similarity between the user preferences and all menu items
    similarity_scores = cosine_similarity(user_vector, all_features)[0]  # Compute cosine similarity
    df['Score'] = similarity_scores  # Assign similarity scores to the dataframe

    # Filter based on the user input (price, rating, etc.)
    filtered_df = df[
        (df["Veg/Non-veg"] == user_input["Veg/Non-veg"]) &
        (df["Price"] <= user_input["Price"]) &
        (df["Food Court/Cafe Location"] == user_input["Food Court/Cafe Location"]) &
        (df["Rating"] >= user_input["Rating"])
    ]

    # Add spice match column
    filtered_df["Spice Match"] = (filtered_df["Spice Level"] == user_input["Spice Level"]).astype(int)

    # Sort recommendations by spice match, predicted rating, and cosine similarity score
    recommendations = filtered_df.sort_values(
        by=["Spice Match", "Predicted_Rating", "Score"],
        ascending=[False, False, False]
    ).drop_duplicates(subset=["Food Item Name"]).head(5)

    return recommendations[['Food Item Name', 'Food Court/Cafe Location', 'Price', 'Meal Type', 'Cuisine Type']]
