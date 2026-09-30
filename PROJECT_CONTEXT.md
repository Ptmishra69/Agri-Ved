# Agri-Ved: Project Context & Architecture

## Overview
**Agri-Ved** is a web-based agricultural platform designed to help farmers and agribusinesses make data-driven decisions. It provides tools for crop recommendations based on soil health, a 7-day weather forecast, and a conversational AI assistant (Chatbot) that offers guidance in multiple languages.

## Project Structure
The project is organized into clean, standard web development folders:

```text
Agri-Ved/
├── assets/
│   └── images/          # Images (logo.png, etc.) used across the UI
├── css/
│   └── style.css        # Vanilla CSS file to supplement Tailwind classes
├── my_rasa_bot/         # (Legacy) Original Rasa implementation and Machine Learning models
│   └── actions/
│       ├── crop_model.joblib   # Trained ML Model for predicting crops
│       └── scaler.joblib       # Feature scaler for the ML model
├── backend.py           # Modern FastAPI backend replacing Rasa (Python 3.12 compatible)
├── index.html           # Main Landing Page
├── chatbot2.htm         # Chatbot User Interface
├── form2.html           # Authentication & Registration UI
└── weather.html         # Weather Dashboard
```

## Features Built So Far

### 1. Frontend Interfaces (HTML/Tailwind CSS/JS)
- **Landing Page (`index.html`)**: Features a responsive hero section, services breakdown (Crop Recommendations, Chatbot, Weather, Fertilizer limits), and a JavaScript-based multi-language switcher supporting English (`en`), Hindi (`hi`), and Punjabi (`pa`).
- **Authentication (`form2.html`)**: Handles Sign In and Sign Up. During Sign Up, it integrates with the OpenStreetMap (Nominatim) API to convert the user's location text into Latitude/Longitude coordinates, which are then saved to `localStorage` for future use. It also collects Soil Health Card (SHC) IDs.
- **Weather Dashboard (`weather.html`)**: Retrieves the user's saved Latitude/Longitude from `localStorage` and fetches a 7-day weather forecast using the OpenWeatherMap API. Uses `Chart.js` to render interactive charts for Temperature, Humidity, and Rainfall.
- **Chatbot UI (`chatbot2.htm`)**: An interactive chat interface that supports both Text input and Voice input (using `webkitSpeechRecognition`). It sends POST requests containing the user's message to the backend via `http://localhost:5005/webhooks/rest/webhook`.

### 2. Backend API & Machine Learning (`backend.py`)
- **Framework**: Built with **FastAPI** to be lightweight, incredibly fast, and fully compatible with modern Python 3.12+ (replacing the heavy Rasa dependency).
- **Rule-Based NLP Engine**: Handles conversational intents via Keyword and Regex matching:
  - Greetings (`hello`, `hi`)
  - Exits (`goodbye`)
  - Intent classification for crop recommendation and SHC requests.
- **Machine Learning Integration**: 
  - Loads a pre-trained scikit-learn model (`crop_model.joblib`) and data scaler (`scaler.joblib`).
  - When a user provides an SHC ID (e.g., `SHC123`), the backend looks up their Nitrogen (N), Phosphorus (P), Potassium (K), pH, and weather factors.
  - The ML model runs a prediction and returns the **best recommended crop** directly in the chat.
  - *(Note: Currently, the SHC data is simulated via a dummy Python dictionary `SHC_DATA` within `backend.py`, which is intended to be replaced with a real government API or Database).*

## How to Run

1. **Frontend**: Open `index.html` in a web browser (or use VS Code Live Server).
2. **Backend**: 
   - Install dependencies: `pip install fastapi uvicorn pandas scikit-learn joblib`
   - Run the server: `python backend.py`
   - The bot runs on `http://localhost:5005`, exactly where the frontend expects it.

## Next Steps / Future Enhancements
- Replace the dummy `SHC_DATA` in `backend.py` with an actual database connection or API call to fetch live soil data.
- Refine the voice-recognition experience to support native translation pipelines.
- Expand the NLP engine in `backend.py` to cover more diverse intents (e.g., specific fertilizer advice or real-time crop pricing).
