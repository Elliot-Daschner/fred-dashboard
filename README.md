# FRED Economic Dashboard

A single-page dashboard for six U.S. macroeconomic indicators, pulled live from the Federal Reserve Economic Data (FRED) API.

**Live:** https://fred-dashboard-5txq.onrender.com/

<img width="1694" height="933" alt="FRED_Dashboard" src="https://github.com/user-attachments/assets/d04f4116-70b3-4e25-86af-585e73b2583c" />

## What it does

Six series, each with its own chart: unemployment, inflation, GDP, the federal funds rate, the 10-year Treasury yield, and consumer sentiment.

- Interactive time-series charts with NBER recession periods shaded
- Date range filtering
- CSV export of the underlying observations
- A 6x6 correlation heatmap across all indicators
- Mobile responsive

## Stack

Python, Flask, and a Jinja template with inline JavaScript for the charts. No frontend build step and no database. Deployed on Render, served by gunicorn via a Procfile.

## How it works

The backend is a thin aggregator over the FRED API rather than a data store. A single `get_fred_data` function calls FRED's observations endpoint and reshapes the response; six routes wrap the six series. Everything is fetched live on page load.

### Parallel fetching for the heatmap

The `/api/all` endpoint feeds the correlation heatmap and needs all six series at once. It originally fetched them sequentially, so the wall-clock load time was the sum of six network round trips. Since the calls are blocking I/O rather than CPU work, a `ThreadPoolExecutor` lets the waits overlap instead of stacking, which cut load time substantially for no added complexity.

### Aligning series that report at different frequencies

This was the actual problem in building the heatmap. The six series report on different schedules: some daily, some monthly, GDP quarterly. Pearson correlation needs paired observations, and these series do not line up on their own.

The heatmap is computed client-side after `/api/all` returns. It builds a lookup per series, aligns them by nearest-date matching, then computes Pearson correlation for all 36 pairs and colors each cell by interpolating an RGB range.

Nearest-date matching is an approximation, and worth naming as one. It does not forward-fill, does not account for the lag between a quarterly figure and the monthly data it is matched against, and produces no confidence bounds on the coefficients. The heatmap is meant to surface rough co-movement between indicators, not to support inference.

### A bug worth documenting

GDP loaded slowly and returned less history than the other series. Two unrelated causes, which is why it took a while to find:

1. The fetch had no timeout, so a slow FRED response blocked the request instead of failing fast.
2. GDP was requested at the same observation limit as the monthly series. Because GDP is quarterly, an identical limit buys roughly a third as many years of history.

Fixed by adding a timeout and raising GDP's observation limit specifically.

## Known limitations

Kept deliberately visible rather than papered over:

- **No caching or rate limiting.** Every page load hits FRED live.
- **Redundant fetches.** The six chart endpoints and `/api/all` request the same six series, roughly doubling FRED calls per load. Known, unaddressed.
- **Recession shading is hardcoded** from NBER dates rather than pulled from a source, so it needs a manual update when NBER announces a new cycle.
- **Correlation alignment is approximate**, as described above.
- **No automated tests.**

## Running locally

```bash
git clone https://github.com/Elliot-Daschner/fred-dashboard.git
cd fred-dashboard
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free API key from https://fredaccount.stlouisfed.org/apikeys and put it in a `.env` file in the project root:

```
FRED_API_KEY=your_key_here
```

The key is loaded with python-dotenv and is never committed.

```bash
python app.py
```

Then open http://localhost:5000.

## Possible next steps

- Cache FRED responses so repeat loads do not refetch identical data
- Serve the charts and the heatmap from one request instead of two
- Replace nearest-date matching with proper resampling before computing correlations
- Add tests around the data-reshaping layer
