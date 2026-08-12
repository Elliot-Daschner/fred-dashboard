from flask import Flask, jsonify, render_template
from fred_client import get_fred_data
import recession_model

app = Flask(__name__)

@app.route("/api/inflation")
def inflation():
    return jsonify(get_fred_data("CPIAUCSL"))

@app.route("/api/unemployment")
def unemployment():
    return jsonify(get_fred_data("UNRATE"))

@app.route("/api/gdp")
def gdp():
    return jsonify(get_fred_data("GDP", limit=100))

@app.route("/api/ffr")
def ffr():
    return jsonify(get_fred_data("FEDFUNDS"))

@app.route("/api/treasury")
def treasury():
    return jsonify(get_fred_data("DGS10"))

@app.route("/api/sentiment")
def sentiment():
    return jsonify(get_fred_data("UMCSENT"))

@app.route("/api/sahm")
def sahm():
    return jsonify(get_fred_data("SAHMREALTIME", limit=800))

@app.route("/api/mortgage")
def mortgage():
    return jsonify(get_fred_data("MORTGAGE30US", limit=600))

@app.route("/api/recession-model")
def recession_model_endpoint():
    try:
        return jsonify(recession_model.predict_current())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recession-watch")
def recession_watch():
    return render_template("recession_watch.html")

@app.route("/api/recessions")
def recessions():
    return jsonify([
        {"start": "1980-01-01", "end": "1980-07-01"},
        {"start": "1981-07-01", "end": "1982-11-01"},
        {"start": "1990-07-01", "end": "1991-03-01"},
        {"start": "2001-03-01", "end": "2001-11-01"},
        {"start": "2007-12-01", "end": "2009-06-01"},
        {"start": "2020-02-01", "end": "2020-04-01"}
    ])

import concurrent.futures

@app.route("/api/all")
def all_series():
    series_ids = {
        "unemployment": "UNRATE",
        "inflation": "CPIAUCSL",
        "gdp": "GDP",
        "ffr": "FEDFUNDS",
        "treasury": "DGS10",
        "sentiment": "UMCSENT"
    }
    
    def fetch(item):
        key, series_id = item
        return key, get_fred_data(series_id)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = dict(executor.map(fetch, series_ids.items()))
    
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)