import os
import joblib
import pandas as pd
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

# Paths for model and scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ML model and scaler loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load ML model/scaler: {e}")

# Dummy SHC dataset (replace with real government API later)
SHC_DATA = {
    "SHC123": {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202},
    "SHC456": {"N": 120, "P": 50, "K": 40, "temperature": 25.0, "humidity": 70, "ph": 7.0, "rainfall": 180},
}

class ActionGetSoilHealth(Action):
    def name(self):
        return "action_process_soil_data"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        
        shc_id = tracker.get_slot("shc_id")  # Get SHC ID from user

        if not shc_id or shc_id not in SHC_DATA:
            dispatcher.utter_message(text="❌ Invalid SHC ID. Please check and try again.")
            return []

        # Get soil data
        features = SHC_DATA[shc_id]

        # Convert to DataFrame with correct column names
        df = pd.DataFrame([features], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])

        # Scale features and predict crop
        X_scaled = scaler.transform(df)
        prediction = model.predict(X_scaled)[0]

        dispatcher.utter_message(
            text=f"🌱 Based on SHC ID `{shc_id}`, the recommended crop is: **{prediction}**"
        )
        return []
