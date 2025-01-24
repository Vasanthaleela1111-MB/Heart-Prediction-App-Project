import pandas as pd
import joblib

# Load the saved model, scaler, and label encoders
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to preprocess new data
def preprocess_data(new_data, label_encoders, scaler):
    # Ensure the input is a DataFrame
    new_data = pd.DataFrame(new_data)
    
    # Encode categorical variables
    categorical_columns = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    for column in categorical_columns:
        le = label_encoders[column]
        new_data[column] = le.transform(new_data[column])
    
    # Scale numerical variables
    new_data_scaled = scaler.transform(new_data)
    return new_data_scaled

# Example new data (replace with actual new data)
new_data = {
    'Age': [55],
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingBP': [140],
    'Cholesterol': [220],
    'FastingBS': [0],
    'RestingECG': ['Normal'],
    'MaxHR': [160],
    'ExerciseAngina': ['N'],
    'Oldpeak': [0.0],
    'ST_Slope': ['Up']
}

# Preprocess the new data
new_data_scaled = preprocess_data(new_data, label_encoders, scaler)

# Make prediction
prediction = model.predict(new_data_scaled)
prediction_proba = model.predict_proba(new_data_scaled)

# Output the prediction
print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Prediction Probability: {prediction_proba[0]}")
