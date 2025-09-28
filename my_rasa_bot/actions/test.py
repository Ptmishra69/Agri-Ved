import joblib
import numpy as np
import os
print(os.getcwd()) 

# Load model and scaler
model = joblib.load("crop_model.joblib")
scaler = joblib.load("scaler.joblib")

# Example input: [N, P, K, temperature, humidity, pH, rainfall]
# Replace with realistic values from your soil/SHC data
sample_input = np.array([[90, 40, 40, 25, 80, 6.5, 200]])

# Scale input
sample_scaled = scaler.transform(sample_input)

# Predict crop
prediction = model.predict(sample_scaled)
print("Predicted Crop:", prediction[0])