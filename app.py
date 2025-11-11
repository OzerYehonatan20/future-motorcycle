from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import joblib
import numpy as np
import pandas as pd
import requests, re
from bs4 import BeautifulSoup


# ==============================
# Load model + dataset
# ==============================
df = pd.read_csv("motorcycles_dataset_merged.csv")
model = joblib.load("motorcycle_model_final.pkl")

app = FastAPI(title="🏍️ Ozer Motor - Future Motorcycle Rating")

# static for background image
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================
# Home page (with background)
# ==============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
      <meta charset='utf-8'>
      <title>🏍️ Ozer Motor</title>
      <link href="https://fonts.googleapis.com/css2?family=Rubik:wght@400;700&display=swap" rel="stylesheet">
      <style>
        body {
          font-family: 'Rubik', sans-serif;
          text-align: center;
          padding: 50px;
          color: white;
          background-image: url('/static/background.png');
          background-size: cover;
          background-position: center;
          background-attachment: fixed;
          position: relative;
        }
        body::before {
          content: "";
          position: fixed;
          top: 0; left: 0; width: 100%; height: 100%;
          background: rgba(0,0,0,0.55);
          z-index: -1;
        }
        h1 { font-size: 2.4em; margin-bottom: 10px; }
        p { font-size: 1.1em; color: #ddd; margin-bottom: 25px; }
        input { width: 300px; padding: 10px; font-size: 16px; border-radius: 8px; border: 1px solid #ccc; margin: 6px; }
        button { padding: 10px 25px; background: #007bff; color: white; border: none; border-radius: 8px; cursor: pointer; font-weight: bold; }
        button:hover { background: #0056b3; }
        #result { margin-top: 25px; font-size: 20px; font-weight: bold; }
        #manual-form { margin-top: 40px; }
      </style>
    </head>
    <body>
      <h1>🏍️ Future Motorcycle Rating</h1>
      <p>Paste a used motorcycle ad URL (Yad2, WinWin, Bikedeals...) or fill manually 👇</p>

      <input id="url" type="text" placeholder="https://www.yad2.co.il/vehicles/item/..." />
      <button onclick="predictFromUrl()">Predict from URL</button>

      <div id="manual-form">
        <h3>🔧 Manual or Missing Data Entry</h3>
        <input id="year" placeholder="Year (e.g. 2023)" /><br>
        <input id="engine_cc" placeholder="Engine CC" /><br>
        <input id="hand" placeholder="Hand (1-5)" /><br>
        <input id="km" placeholder="Kilometers" /><br>
        <input id="price" placeholder="Price (₪)" /><br>
        <button onclick="predictManual()">Predict Manually</button>
      </div>

      <div id="result"></div>

      <script>
        async function predictFromUrl() {
          const url = document.getElementById('url').value;
          document.getElementById('result').innerText = "⏳ Analyzing...";
          try {
            const response = await fetch(`/predict/url?link=${encodeURIComponent(url)}`);
            const data = await response.json();
            if (data.error) {
              document.getElementById('result').innerText = "⚠️ " + data.error;
            } else {
              document.getElementById('result').innerText = `⭐ Predicted Rating: ${data.predicted_rating}/10`;
            }
          } catch (err) {
            document.getElementById('result').innerText = "❌ Error contacting server.";
          }
        }

        async function predictManual() {
          const year = document.getElementById('year').value;
          const engine_cc = document.getElementById('engine_cc').value;
          const hand = document.getElementById('hand').value;
          const km = document.getElementById('km').value;
          const price = document.getElementById('price').value;
          document.getElementById('result').innerText = "⏳ Calculating...";
          const response = await fetch(`/predict/manual?year=${year}&engine_cc=${engine_cc}&hand=${hand}&km=${km}&price=${price}`);
          const data = await response.json();
          if (data.error) {
            document.getElementById('result').innerText = "⚠️ " + data.error;
          } else {
            document.getElementById('result').innerText = `⭐ Predicted Rating: ${data.predicted_rating}/10`;
          }
        }
      </script>
    </body>
    </html>
    """


# ==============================
# Core prediction logic
# ==============================
def predict_rating(year, engine_cc, hand, km, price):
    age = 2025 - year
    km_per_year = km / max(1, age)
    price_per_cc = price / engine_cc
    price_per_year = price / max(1, age)
    normalized_price = price / df["price"].max()
    log_km = np.log1p(km)
    log_price = np.log1p(price)

    data = np.array([[age, engine_cc, hand, km, price,
                      km_per_year, price_per_cc, price_per_year,
                      normalized_price, log_km, log_price]])
    rating = float(model.predict(data)[0])
    return round(rating, 2)


# ==============================
# URL scraping + extraction
# ==============================
def extract_data_from_url(url):
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r'[\u200f\u200e]', '', text)
    text = re.sub(r'\s+', ' ', text)

    # מחיר
    price_match = re.search(r'([\d,]+)\s*₪', text)
    price = int(price_match.group(1).replace(',', '')) if price_match else None

    # שנה
    year_match = re.search(r'(20\d{2})', text)
    year = int(year_match.group(1)) if year_match else None

    # קילומטרים
    km_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*ק', text)
    km = int(km_match.group(1).replace(',', '')) if km_match else None

    # סמ"ק
    cc_match = re.search(r'נפח מנוע[:\s]*(\d{2,4})', text) or re.search(r'(\d{2,4})\s*סמ', text)
    engine_cc = int(cc_match.group(1)) if cc_match else None

    # יד
    hand_match = re.search(r'יד\s*(\d)', text)
    hand = int(hand_match.group(1)) if hand_match else 2

    return year, engine_cc, hand, km, price


# ==============================
# API Endpoints
# ==============================
@app.get("/predict/url")
def predict_from_url(link: str = Query(..., description="Motorcycle ad URL")):
    try:
        year, engine_cc, hand, km, price = extract_data_from_url(link)
        if None in [year, engine_cc, hand, km, price]:
            return JSONResponse({"error": "Could not extract all fields. Try manual fill."})
        rating = predict_rating(year, engine_cc, hand, km, price)
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": f"Error: {str(e)}"})


@app.get("/predict/manual")
def predict_manual(year: int, engine_cc: int, hand: int, km: int, price: int):
    try:
        rating = predict_rating(year, engine_cc, hand, km, price)
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": str(e)})
