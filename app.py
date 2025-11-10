from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

# Load dataset (for normalization)
df = pd.read_csv("motorcycles_dataset_merged.csv")

# Load model
model = joblib.load("motorcycle_model_final.pkl")

app = FastAPI(title="🏍️ Future Motorcycle Rating API")

class MotorcycleInput(BaseModel):
    year: int
    engine_cc: int
    hand: int
    km: float
    price: float

@app.post("/predict")
def predict_rating(data: MotorcycleInput):
    age = 2025 - data.year
    km_per_year = data.km / max(1, age)
    price_per_cc = data.price / data.engine_cc
    price_per_year = data.price / max(1, age)
    normalized_price = data.price / df["price"].max()
    log_km = np.log1p(data.km)
    log_price = np.log1p(data.price)

    features = np.array([[age, data.engine_cc, data.hand, data.km, data.price,
                          km_per_year, price_per_cc, price_per_year,
                          normalized_price, log_km, log_price]])

    rating = float(model.predict(features)[0])
    return {"predicted_rating": round(rating, 2)}

@app.get("/")
def root():
    return {"status": "ok", "message": "Future Motorcycle API is live 🚀"}
