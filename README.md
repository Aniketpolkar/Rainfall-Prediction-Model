# 🌧️ Rainfall Prediction App for Northeast India

This is a Streamlit-based web application that predicts **rainfall (precipitation)** using historical weather data. It allows users to input feature values and get predicted rainfall using the best performing machine learning model.

## 🔍 Features
- Interactive UI with **Streamlit**
- Supports input of custom weather features
- Uses **Random Forest**, **SVM**, and **XGBoost** for prediction
- Automatically selects the **best model** based on MAE (Mean Absolute Error)
- Displays **model performance metrics**
- Real-time rainfall prediction based on user inputs

## 📁 Dataset
The model is trained on historical weather data from **Dhubri, Assam**. File used: `Dhuburi.csv`  
Make sure the dataset is placed in the project folder.

## ⚙️ Technologies Used
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib (optional for EDA)
- StandardScaler (for data normalization)
