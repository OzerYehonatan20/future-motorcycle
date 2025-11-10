from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import joblib
import numpy as np
import pandas as pd
import requests, re
from bs4 import BeautifulSoup

# ==============================
# Load model and dataset
# ==============================
df = pd.read_csv("motorcycles_dataset_merged.csv")
model = joblib.load("motorcycle_model_final.pkl")

app = FastAPI(title="🏍️ Future Motorcycle Rating API (Universal URL Version)")

# ==============================
# Homepage (HTML UI)
# ==============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
      <meta charset='utf-8'>
      <title>🏍️ Future Motorcycle Rating</title>
      <style>
        body { 
          font-family: Arial, sans-serif; 
          text-align:center; 
          padding:60px; 
          background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        .container {
          background:white;
          display:inline-block;
          padding:40px 50px;
          border-radius:14px;
          box-shadow:0 6px 15px rgba(0,0,0,0.1);
          transition: all 0.2s ease-in-out;
        }
        .container:hover {
          box-shadow:0 8px 20px rgba(0,0,0,0.15);
        }
        h1 {
          font-size:34px;
          color:#222;
          margin-bottom:10px;
        }
        h1 span {
          font-size:40px;
        }
        p.subtitle {
          font-size:16px;
          color:#555;
          margin-top:0;
          margin-bottom:25px;
        }
        input {
          width:420px;
          padding:10px;
          font-size:16px;
          border-radius:8px;
          border:1px solid #bbb;
        }
        button {
          padding:10px 25px;
          background:#007bff;
          color:white;
          border:none;
          border-radius:8px;
          cursor:pointer;
          font-size:16px;
          margin-top:10px;
        }
        button:hover { 
          background:#0056b3; 
        }
        button[title]:hover::after {
          content: attr(title);
          position: absolute;
          background: #333;
          color: white;
          padding: 5px 10px;
          border-radius: 5px;
          font-size: 12px;
          top: 60px;
          left: 50%;
          transform: translateX(-50%);
        }
        #result {
          margin-top:25px;
          font-size:20px;
          font-weight:bold;
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1><span>🏍️</span> Future Motorcycle Rating</h1>
        <p class="subtitle">
          Paste here any <b>motorcycle listing URL</b> (Yad2 / Bikedeals / WinWin...) <br>
          and I’ll tell you what it’s really worth 😉
        </p>
        <input id="url" type="text" placeholder="https://www.yad2.co.il/item/..." />
        <br>
        <button onclick="predict()" title="Click to predict rating">🔥 Rate This Ride</button>
        <div id="result"></div>
      </div>

      <script>
        async function predict() {
          const url = document.getElementById('url').value;
          document.getElementById('result').innerText = "⏳ Analyzing...";
          const response = await fetch(`/predict_url?url=${encodeURIComponent(url)}`);
          const data = await response.json();
          if (data.error) {
            document.getElementById('result').innerText = "⚠️ " + data.error;
          } else {
            document.getElementById('result').innerText =
              `⭐ Predicted Rating: ${data.predicted_rating}/10`;
          }
        }
      </script>
    </body>
    </html>
    """

# ==============================
# Prediction logic
# ==============================
def predict_rating(year, engine_cc, hand, km, price):
    age = 2025 - year
    km_per_year = km / max(1, age)
    price_per_cc = price / engine_cc
    price_per_year = price / max(1, age)
    normalized_price = price / df["price"].max()
    log_km = np.log1p(km)
    log_price = np.log1p(price)

    data = np.array([[age, engine_cc, hand, km, price, km_per_year,
                      price_per_cc, price_per_year, normalized_price, log_km, log_price]])
    return round(float(model.predict(data)[0]), 2)

# ==============================
# URL scraping + prediction
# ==============================
def extract_data_from_url(url):
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, "html.parser")

    text = soup.get_text(" ", strip=True)
    text = re.sub(r'[\u200f\u200e]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("״", '"').replace("׳", "'").replace('.', ',')

    # --- PRICE ---
    price_match = re.search(r'מחיר[:\s]*([\d,]+)\s*₪?', text) or re.search(r'([\d,]+)\s*₪', text)
    price = int(price_match.group(1).replace(',', '')) if price_match else None

    # --- YEAR ---
    year_match = re.search(r'שנה[:\s]*(20\d{2})', text) or re.search(r'(20\d{2})', text)
    year = int(year_match.group(1)) if year_match else None

    # --- KILOMETERS ---
    km_match = re.search(r'(?:ק.?ילומ.?|מרחק|נסיעה)[:\s]*(\d{1,3}(?:,\d{3})*)', text) or re.search(r'(\d{1,3}(?:,\d{3})*)\s*(?:ק.?\"?מ)', text)
    km = int(km_match.group(1).replace(',', '')) if km_match else None

    # --- ENGINE CC ---
    cc_match = re.search(r'נפח\s*מנוע[:\s]*(\d{2,4})', text) or re.search(r'(\d{2,4})\s*סמ', text)
    engine_cc = int(cc_match.group(1)) if cc_match else None

    # --- HAND (ownership) ---
    hand_match = re.search(r'יד\s*(\d)', text)
    hand = int(hand_match.group(1)) if hand_match else 2

    if not price:
        price_tag = soup.find(class_=re.compile("price|מחיר"))
        if price_tag:
            price = int(re.sub(r'\D', '', price_tag.text))

    # --- Sanity checks ---
    if year and not (2000 <= year <= 2025): year = None
    if km and km < 100: km = None
    if price and price < 2000: price = None

    return year, engine_cc, hand, km, price


@app.get("/predict_url")
def predict_from_url(url: str = Query(..., description="Motorcycle ad URL")):
    try:
        year, engine_cc, hand, km, price = extract_data_from_url(url)
        if None in [year, engine_cc, hand, km, price]:
            return JSONResponse({"error": "❌ Could not extract all fields from the link."})
        rating = predict_rating(year, engine_cc, hand, km, price)
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": f"Error: {str(e)}"})
