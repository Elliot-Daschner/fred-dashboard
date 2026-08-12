import os
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def get_fred_data(series_id, limit=50):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc"
    }
    response = requests.get(FRED_BASE_URL, params=params, timeout=10)
    data = response.json()
    observations = data.get("observations", [])
    return [{"date": o["date"], "value": o["value"]} for o in observations]
