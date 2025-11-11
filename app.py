from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

app = FastAPI(title="🏍️ Ozer Motor — Future Motorcycle Rating")

# Allow static files (for background)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ==============================
# Homepage with background
# ==============================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
      <meta charset='utf-8'>
      <title>🏍️ Ozer Motor</title>
      <link href="https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap" rel="stylesheet">
      <style>
        body {
            font-family: 'Segoe UI', Arial;
            text-align:center;
            padding:60px;
            color:white;
            background-image: url('/static/background.png');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        body::before {
            content: "";
            position: fixed;
            top:0; left:0;
            width:100%; height:100%;
            background: rgba(0,0,0,0.55);
            z-index:-1;
        }
        h1 {
            font-family: 'Press Start 2P', cursive;
            font-size: 36px;
            margin-bottom: 10px;
            color: #fff;
            text-shadow: 2px 2px 4px #000;
        }
        p {
            font-size: 1.1em;
            color: #ddd;
            margin-bottom: 20px;
        }
        input, button {
            width: 320px;
            padding:10px;
            font-size:16px;
            border-radius:8px;
            border:1px solid #ccc;
            margin:6px;
        }
        button {
            background:#007bff;
            color:white;
            cursor:pointer;
            border:none;
            box-shadow:0 0 10px rgba(0,0,0,0.4);
        }
        button:hover { background:#0056b3; }
        #manual-form { display:none; margin-top:30px; }
        #result { margin-top:25px; font-size:20px; font-weight:bold; }
      </style>
    </head>
    <body>
      <h1>🏍️ Ozer Motor</h1>
      <p>Paste a used motorcycle ad URL (Yad2, WinWin, Bikedeals...) or fill manually 👇</p>
      <input id="url" type="text" placeholder="https://..." />
      <button onclick="predict()">Predict from URL</button>

      <div id="manual-form">
        <h3>🔧 Manual or Missing Data Entry</h3>
        <input id="year" placeholder="Year (e.g. 2018)" /><br>
        <input id="cc" placeholder="Engine CC" /><br>
        <input id="hand" placeholder="Hand (1-5)" /><br>
        <input id="km" placeholder="Kilometers" /><br>
        <input id="price" placeholder="Price (₪)" /><br>
        <button onclick="manualPredict()">Predict Manually</button>
      </div>

      <div id="result"></div>

      <script>
        async function predict() {
          const url = document.getElementById('url').value;
          document.getElementById('result').innerText = "⏳ Scraping...";
          const response = await fetch(`/predict/url?link=${encodeURIComponent(url)}`);
          const data = await response.json();

          if (data.error && data.error.includes("extract")) {
            document.getElementById('result').innerText = "⚠️ Missing data — please complete manually below.";
            document.getElementById('manual-form').style.display = "block";
          } else if (data.error) {
            document.getElementById('result').innerText = "❌ " + data.error;
          } else {
            document.getElementById('result').innerHTML = `⭐ Predicted Rating: <b>${data.predicted_rating}/10</b>`;
          }
        }

        async function manualPredict() {
          const params = new URLSearchParams({
            year: document.getElementById('year').value,
            engine_cc: document.getElementById('cc').value,
            hand: document.getElementById('hand').value,
            km: document.getElementById('km').value,
            price: document.getElementById('price').value
          });
          document.getElementById('result').innerText = "⚙️ Calculating...";
          const response = await fetch(`/predict/manual?${params}`);
          const data = await response.json();
          if (data.error) {
            document.getElementById('result').innerText = "⚠️ " + data.error;
          } else {
            document.getElementById('result').innerHTML = `⭐ Predicted Rating: <b>${data.predicted_rating}/10</b>`;
          }
        }
      </script>
    </body>
    </html>
    """


# ==============================
# Prediction Logic
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
# Scraping logic
# ==============================
def extract_data_from_url(url):
    res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
    soup = BeautifulSoup(res.text, "html.parser")
    text = soup.get_text(" ", strip=True)
    text = re.sub(r'\s+', ' ', text)

    price_match = re.search(r'([\d,]+)\s*₪', text)
    year_match = re.search(r'(20\d{2})', text)
    km_match = re.search(r'(\d{1,3}(?:,\d{3})*)\s*(?:ק.?\"?מ)', text)
    cc_match = re.search(r'(\d{2,4})\s*סמ', text)
    hand_match = re.search(r'יד\s*(\d)', text)

    price = int(price_match.group(1).replace(',', '')) if price_match else None
    year = int(year_match.group(1)) if year_match else None
    km = int(km_match.group(1).replace(',', '')) if km_match else None
    engine_cc = int(cc_match.group(1)) if cc_match else None
    hand = int(hand_match.group(1)) if hand_match else 2

    return year, engine_cc, hand, km, price


# ==============================
# Endpoints
# ==============================
@app.get("/predict/url")
def predict_from_url(link: str = Query(...)):
    try:
        year, engine_cc, hand, km, price = extract_data_from_url(link)
        if None in [year, engine_cc, hand, km, price]:
            return JSONResponse({"error": "Could not extract all fields from the link."})
        rating = predict_rating(year, engine_cc, hand, km, price)
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": str(e)})


@app.get("/predict/manual")
def predict_manual(year: int, engine_cc: int, hand: int, km: int, price: int):
    try:
        rating = predict_rating(year, engine_cc, hand, km, price)
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": str(e)})
