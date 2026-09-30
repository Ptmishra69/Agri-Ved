import re
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import httpx
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import jwt

app = FastAPI()

# ──────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# JWT Configuration
# ──────────────────────────────────────────────
SECRET_KEY = secrets.token_hex(32)  # Generated once per server start
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

security = HTTPBearer()


def hash_password(password: str) -> str:
    """Hash a password with SHA-256 + salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored salt:hash."""
    salt, hashed = stored.split(":")
    return hashlib.sha256((salt + password).encode()).hexdigest() == hashed


def create_access_token(data: dict) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Dependency: extract and validate the current user from a Bearer token."""
    payload = decode_token(credentials.credentials)
    phone = payload.get("sub")
    if phone is None or phone not in USERS_DB:
        raise HTTPException(status_code=401, detail="User not found")
    user = USERS_DB[phone].copy()
    user.pop("password", None)  # Never return the password hash
    return user


# ──────────────────────────────────────────────
# Simple JSON-file User Database
# ──────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "users_db.json")

def load_users() -> dict:
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users():
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(USERS_DB, f, indent=2, ensure_ascii=False)

USERS_DB: dict = load_users()

# ──────────────────────────────────────────────
# Indian States List (matching Soil Health Card portal)
# ──────────────────────────────────────────────
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi", "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
]

# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class SignUpRequest(BaseModel):
    name: str
    phone: str
    password: str
    state: str
    district: str
    block: str | None = None
    village: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    shc_id: str | None = None

class SignInRequest(BaseModel):
    phone: str
    password: str

class ChatMessage(BaseModel):
    sender: str
    message: str

# ──────────────────────────────────────────────
# Location API Endpoints (State → District → Block → Village)
# ──────────────────────────────────────────────
@app.get("/api/location/states")
async def get_states():
    """Return all Indian states/UTs."""
    return [{"name": s} for s in INDIAN_STATES]


@app.get("/api/location/districts")
async def get_districts(state: str):
    """Fetch districts for a given state using the Nominatim geocoding API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": f"district, {state}, India", "format": "json", "addressdetails": "1", "limit": "50"},
                headers={"User-Agent": "AgriVed/1.0"}
            )
            data = res.json()
            districts = set()
            for item in data:
                addr = item.get("address", {})
                dist = addr.get("county") or addr.get("state_district") or item.get("display_name", "").split(",")[0]
                if dist:
                    districts.add(dist.strip())
            return [{"name": d} for d in sorted(districts)]
    except Exception:
        return [{"name": "Could not load districts — please type manually"}]


# ──────────────────────────────────────────────
# Auth Endpoints
# ──────────────────────────────────────────────
@app.post("/api/auth/signup")
async def signup(req: SignUpRequest):
    if req.phone in USERS_DB:
        raise HTTPException(status_code=400, detail="Phone number already registered")

    USERS_DB[req.phone] = {
        "name": req.name,
        "phone": req.phone,
        "password": hash_password(req.password),
        "state": req.state,
        "district": req.district,
        "block": req.block,
        "village": req.village,
        "latitude": req.latitude,
        "longitude": req.longitude,
        "shc_id": req.shc_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_users()

    return {"message": "Account created successfully!"}


@app.post("/api/auth/login")
async def login(req: SignInRequest):
    user = USERS_DB.get(req.phone)
    if not user or not verify_password(req.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid phone number or password")

    token = create_access_token({"sub": req.phone, "name": user["name"]})

    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": user["name"],
            "phone": user["phone"],
            "state": user.get("state", ""),
            "district": user.get("district", ""),
            "block": user.get("block"),
            "village": user.get("village"),
            "latitude": user.get("latitude"),
            "longitude": user.get("longitude"),
            "shc_id": user.get("shc_id"),
        }
    }


@app.get("/api/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


# ──────────────────────────────────────────────
# ML Model Loading
# ──────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(__file__), "my_rasa_bot", "actions")
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.joblib")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ ML model and scaler loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load ML model/scaler: {e}")

# Dummy SHC Dataset
SHC_DATA = {
    "SHC123": {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202},
    "SHC456": {"N": 120, "P": 50, "K": 40, "temperature": 25.0, "humidity": 70, "ph": 7.0, "rainfall": 180},
}


# ──────────────────────────────────────────────
# Chatbot NLP Engine
# ──────────────────────────────────────────────
def process_message(msg: str) -> str:
    msg_lower = msg.lower()

    # Intent: Provide SHC ID
    shc_match = re.search(r'shc\d+', msg, re.IGNORECASE)
    if shc_match:
        shc_id = shc_match.group(0).upper()
        if shc_id in SHC_DATA:
            features = SHC_DATA[shc_id]
            df = pd.DataFrame([features], columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"])
            X_scaled = scaler.transform(df)
            prediction = model.predict(X_scaled)[0]
            return f"🌱 Based on SHC ID `{shc_id}`, the recommended crop is: **{prediction}**"
        else:
            return "❌ I couldn't find your SHC details. Please check your ID and try again."

    # Intent: Greeting
    if any(word in msg_lower for word in ["hello", "hi", "hey", "good morning"]):
        return "Hey! 👋 I am AgriBot 🌱. How can I assist you today?"

    # Intent: Goodbye
    if any(word in msg_lower for word in ["bye", "goodbye"]):
        return "Goodbye 👋. Wishing you a great harvest!"

    # Intent: Bot Challenge
    if "are you a bot" in msg_lower or "who are you" in msg_lower:
        return "I am an AI assistant powered by Python 🤖."

    # Intent: Soil Health or Crop Prediction
    if any(word in msg_lower for word in ["soil", "crop", "recommendation", "suggest"]):
        return "Please provide your Soil Health Card (SHC) ID so I can recommend the best crop for you. (e.g. SHC123)"

    # Fallback
    return "I'm not sure how to help with that. Try asking for a crop recommendation or providing your SHC ID."


# ──────────────────────────────────────────────
# Chatbot Endpoint (protected — requires login)
# ──────────────────────────────────────────────
@app.post("/webhooks/rest/webhook")
async def rasa_webhook(chat: ChatMessage, current_user: dict = Depends(get_current_user)):
    response_text = process_message(chat.message)
    return [{"text": response_text}]


if __name__ == "__main__":
    print("🚀 Starting AgriBot API on http://localhost:5005")
    uvicorn.run(app, host="0.0.0.0", port=5005)
