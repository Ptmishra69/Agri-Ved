# actions.py
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import joblib
import numpy as np

# Load model + scaler once (not in every request)
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "crop_model.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.joblib")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


# Example SHC dictionary (replace with API fetch)
shc_data = {
    "SHC123": {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202},
    "SHC456": {"N": 110, "P": 35, "K": 40, "temperature": 23.5, "humidity": 70, "ph": 6.8, "rainfall": 190},
}

class ActionRecommendCrop(Action):
    def name(self):
        return "action_recommend_crop"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: dict):
        
        # 1. Get SHC ID from user message slot/entity
        shc_id = tracker.get_slot("shc_id")  # You must define slot "shc_id" in domain.yml
        
        if not shc_id or shc_id not in shc_data:
            dispatcher.utter_message(text="❌ I couldn't find data for your SHC ID. Please check again.")
            return []

        # 2. Fetch NPK & other values
        data = shc_data[shc_id]
        sample = np.array([[data["N"], data["P"], data["K"], 
                            data["temperature"], data["humidity"], 
                            data["ph"], data["rainfall"]]])
        
        # 3. Scale + Predict
        sample_scaled = scaler.transform(sample)
        prediction = model.predict(sample_scaled)[0]
        
        # 4. Send back result
        dispatcher.utter_message(
            text=f"🌱 Based on your soil health card ({shc_id}), the recommended crop is: **{prediction}**"
        )

        return []

