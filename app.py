from fastapi import FastAPI, Query, Form
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
model = joblib.load("motorcycle_model_final_render.pkl")

app = FastAPI(title="🏍️ Future Motorcycle Rating API (Enhanced Hybrid Version)")


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
            font-family: 'Segoe UI', Arial; 
            text-align:center; 
            padding:60px; 
            background:#f5f5f5; 
            color:#222; 
        }
        h1 { font-size:2.2em; margin-bottom:10px; }
        p { font-size:1.1em; color:#444; margin-bottom:20px; }
        input, select { width:220px; padding:8px; font-size:15px; border-radius:8px; border:1px solid #ccc; margin:5px; }
        button { padding:10px 25px; background:#007bff; color:white; border:none; border-radius:8px; cursor:pointer; }
        button:hover { background:#0056b3; }
        #result { margin-top:25px; font-size:20px; font-weight:bold; }
        #manualForm { display:none; margin-top:30px; }
      </style>
    </head>
    <body>
      <h1>🏍️ Future Motorcycle Rating</h1>
      <p>Paste a used motorcycle ad URL (Yad2, WinWin, Bikedeals...) or fill manually 👇</p>

      <input id="url" type="text" placeholder="https://..." />
      <button onclick="predictFromURL()">Predict from URL</button>

      <div id="manualForm">
        <h3>🔧 Manual or Missing Data Entry</h3>
        <input id="year" type="number" placeholder="Year (e.g. 2018)"><br>
        <input id="engine_cc" type="number" placeholder="Engine CC"><br>
        <input id="hand" type="number" placeholder="Hand (1-5)"><br>
        <input id="km" type="number" placeholder="Kilometers"><br>
        <input id="price" type="number" placeholder="Price (₪)"><br>
        <button onclick="predictManual()">Predict Manually</button>
      </div>

      <div id="result"></div>

      <script>
        async function predictFromURL() {
          const link = document.getElementById('url').value;
          document.getElementById('result').innerText = "⏳ Scraping...";
          const response = await fetch(`/predict/url?link=${encodeURIComponent(link)}`);
          const data = await response.json();

          if (data.error) {
            document.getElementById('result').innerText = "⚠️ " + data.error;
            document.getElementById('manualForm').style.display = 'block';
          } else if (data.partial) {
            document.getElementById('result').innerText = "⚙️ Partial data found — please complete missing fields:";
            document.getElementById('manualForm').style.display = 'block';
            if (data.partial.year) document.getElementById('year').value = data.partial.year;
            if (data.partial.engine_cc) document.getElementById('engine_cc').value = data.partial.engine_cc;
            if (data.partial.hand) document.getElementById('hand').value = data.partial.hand;
            if (data.partial.km) document.getElementById('km').value = data.partial.km;
            if (data.partial.price) document.getElementById('price').value = data.partial.price;
          } else {
            document.getElementById('result').innerText = `⭐ Predicted Rating: ${data.predicted_rating}/10`;
            document.getElementById('manualForm').style.display = 'none';
          }
        }

        async function predictManual() {
          const payload = {
            year: parseInt(document.getElementById('year').value),
            engine_cc: parseInt(document.getElementById('engine_cc').value),
            hand: parseInt(document.getElementById('hand').value),
            km: parseInt(document.getElementById('km').value),
            price: parseInt(document.getElementById('price').value)
          };
          document.getElementById('result').innerText = "⚙️ Predicting...";
          const res = await fetch('/predict/manual', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          document.getElementById('result').innerText = `⭐ Predicted Rating: ${data.predicted_rating}/10`;
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
    data = np.array([[age, engine_cc, hand, km, price, km_per_year,
                      price_per_cc, price_per_year, normalized_price, log_km, log_price]])
    return round(float(np.clip(model.predict(data)[0], 0, 10)), 2)


# ==============================
# URL scraping + extraction
# ==============================
def extract_data_from_url(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        text = soup.get_text(" ", strip=True)
        text = re.sub(r'[\u200f\u200e]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.replace("״", '"').replace("׳", "'").replace('.', ',')

        def extract(patterns):
            for p in patterns:
                m = re.search(p, text)
                if m:
                    return int(re.sub(r'\D', '', m.group(1)))
            return None

        price = extract([r'מחיר[:\s]*([\d,]+)\s*₪?', r'([\d,]+)\s*₪'])
        year = extract([r'שנה[:\s]*(20\d{2})'])
        km = extract([r'(?:ק.?ילומ.?|מרחק|נסיעה)[:\s]*(\d{1,3}(?:,\d{3})*)', r'(\d{1,3}(?:,\d{3})*)\s*(?:ק.?\"?מ)'])
        engine_cc = extract([r'נפח\s*מנוע[:\s]*(\d{2,4})', r'(\d{2,4})\s*סמ'])
        hand_match = re.search(r'יד\s*(\d)', text)
        hand = int(hand_match.group(1)) if hand_match else None

        partial_data = dict(year=year, engine_cc=engine_cc, hand=hand, km=km, price=price)
        missing = [k for k, v in partial_data.items() if v is None]
        if any(missing) and any(v is not None for v in partial_data.values()):
            return {"partial": partial_data}
        elif all(v is not None for v in partial_data.values()):
            return {"complete": partial_data}
        else:
            return {"error": "No valid fields found."}

    except Exception as e:
        return {"error": f"Scraping failed: {e}"}


# ==============================
# Prediction endpoints
# ==============================
@app.get("/predict/url")
def predict_from_url(link: str = Query(..., description="Motorcycle ad URL")):
    result = extract_data_from_url(link)
    if "error" in result:
        return JSONResponse({"error": result["error"]})
    if "partial" in result:
        return JSONResponse({"partial": result["partial"]})
    if "complete" in result:
        d = result["complete"]
        rating = predict_rating(d["year"], d["engine_cc"], d["hand"], d["km"], d["price"])
        return {"predicted_rating": rating}


@app.post("/predict/manual")
def predict_manual(data: dict):
    try:
        rating = predict_rating(data["year"], data["engine_cc"], data["hand"], data["km"], data["price"])
        return {"predicted_rating": rating}
    except Exception as e:
        return JSONResponse({"error": str(e)})


# ==============================
# Local run
# ==============================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
