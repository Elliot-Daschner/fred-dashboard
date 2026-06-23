from flask import Flask, jsonify, render_template
from flask import Flask, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def get_fred_data(series_id):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": 50,
        "sort_order": "desc"
    }
    response = requests.get(FRED_BASE_URL, params=params)
    data = response.json()
    observations = data.get("observations", [])
    return [{"date": o["date"], "value": o["value"]} for o in observations]

@app.route("/api/inflation")
def inflation():
    return jsonify(get_fred_data("CPIAUCSL"))

@app.route("/api/unemployment")
def unemployment():
    return jsonify(get_fred_data("UNRATE"))

@app.route("/api/gdp")
def gdp():
    return jsonify(get_fred_data("GDP"))

@app.route("/api/ffr")
def ffr():
    return jsonify(get_fred_data("FEDFUNDS"))

@app.route("/api/treasury")
def treasury():
    return jsonify(get_fred_data("DGS10"))

@app.route("/api/sentiment")
def sentiment():
    return jsonify(get_fred_data("UMCSENT"))

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)