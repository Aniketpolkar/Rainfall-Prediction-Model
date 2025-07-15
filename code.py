import pandas as pd
import numpy as np
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error   

# Ignore warnings
warnings.filterwarnings("ignore")

# Streamlit UI
st.title("Rainfall Prediction App For NorthEast India")
st.write("Enter the required feature values to predict rainfall (precipitation).")

st.image("https://s.w-x.co/in-northeast_rain_october_19-21.jpg", caption="Rainfall Map")
@st.cache_data
def load_data():
    """Load dataset and preprocess it."""
    file_path = "Dhuburi.csv"  # Update this with the correct path
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    columns_to_remove = ["Unnamed: 0", "date", "snowfall", "snow_depth"]
    df = df.drop(columns=[col for col in columns_to_remove if col in df.columns])

    return df

@st.cache_resource
def train_models(X_train_scaled, y_train, X_test_scaled, y_test):
    """Train models and return the best one."""
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "SVM": SVR(kernel="rbf"),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
    }

    results = {}
    best_model = None
    best_mae = float("inf")

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate errors
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {"MAE": mae, "MSE": mse}
        
        # Select the best model based on MAE
        if mae < best_mae:
            best_mae = mae
            best_model = model

    return best_model, results

# Load data
df = load_data()

# Define features and target variable
X = df.drop(columns=["precipitation"])  # Features
y = df["precipitation"]  # Target

# Handle missing values
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and get the best model
best_model, results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)

# User input form
user_input = {}
for feature in X.columns:
    user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0, format="%.2f")

if st.button("Predict Rainfall"):
    user_data = pd.DataFrame([user_input])
    user_data_scaled = scaler.transform(user_data)
    predicted_rainfall = best_model.predict(user_data_scaled)
    st.write(f"### Predicted Rainfall (Precipitation): {predicted_rainfall[0]:.4f} mm")

# Display model performance
st.subheader("Model Performance")
for model_name, metrics in results.items():
    st.write(f"**Model: {model_name}**")
    st.write(f"   - MAE: {metrics['MAE']:.4f}")
    st.write(f"   - MSE: {metrics['MSE']:.4f}")
