import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY")
FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

def get_fred_data(series_id, limit=50, retries=2, timeout=20):
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "limit": limit,
        "sort_order": "desc"
    }
    # timeout raised from 10s to 20s, plus one retry on transient network
    # errors (timeouts, connection resets) -- added after the recession
    # model's endpoint, which fires 7 concurrent requests at FRED per
    # prediction, occasionally hit a read timeout on production when one
    # of those 7 was momentarily slow. A single flaky request used to
    # fail the whole batch; this gives it one more chance before giving
    # up. Only genuine network-level errors are retried -- a bug in
    # response handling wouldn't be fixed by retrying, so those still
    # propagate immediately.
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(FRED_BASE_URL, params=params, timeout=timeout)
            data = response.json()
            observations = data.get("observations", [])
            return [{"date": o["date"], "value": o["value"]} for o in observations]
        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < retries - 1:
                time.sleep(1)
    raise last_error
