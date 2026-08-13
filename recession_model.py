"""
Recession probability model.

Estimates the probability of a U.S. recession starting within the next 12
months, using a logistic regression fit on historical FRED data. See the
project's implementation plan / README for the full explanation of why
logistic regression, why these features, and why there's no persisted
model file or accuracy score.

No persisted model artifact: the model is trained once, lazily, on first
use, and cached in memory for the process's lifetime -- the same pattern
dashboard.js's `recessionData` uses on the frontend (fetch once, reuse).
"""

import concurrent.futures
import threading
import warnings
from datetime import datetime, timezone

import pandas as pd
from scipy.optimize import OptimizeWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from fred_client import get_fred_data

# scikit-learn 1.6.1's lbfgs solver wrapper passes a verbosity option
# ("iprint") through to scipy.optimize.minimize that this scipy version
# doesn't recognize -- cosmetic only (the fit still converges normally,
# verified via model.n_iter_ during development), so it's suppressed
# here rather than left to clutter server logs on every training call.
warnings.filterwarnings("ignore", category=OptimizeWarning, message="Unknown solver options")

# FRED series used to build the feature matrix + label.
# GS10/TB3MS (not the dashboard's daily DGS10) so everything is already
# monthly -- no daily-to-monthly resampling needed.
FRED_SERIES = {
    "yield_long": "GS10",        # 10-Year Treasury, monthly constant maturity
    "yield_short": "TB3MS",      # 3-Month T-Bill, monthly secondary market
    "unrate": "UNRATE",
    "sahm": "SAHMREALTIME",
    "sentiment": "UMCSENT",
    "cpi": "CPIAUCSL",
    "label_source": "USREC",     # NBER recession indicator, 0/1, monthly
}

FEATURES = ["yield_spread", "sahm", "sentiment", "cpi_yoy"]

FEATURE_LABELS = {
    "yield_spread": "10Y minus 3M Treasury Spread",
    "sahm": "Sahm Rule Value",
    "sentiment": "Consumer Sentiment Index",
    "cpi_yoy": "CPI Inflation, Year-over-Year %",
}

# unrate_chg_12m was dropped from FEATURES: it's 84% correlated with sahm
# (the Sahm Rule is itself built from short-term unemployment-rate
# dynamics, so the two are largely redundant). With both in the model,
# they fought over the same shared variance and sahm's coefficient came
# out with a sign that looked backwards -- true on its own (univariate,
# sahm's coefficient is positive as expected), but confusing to read in
# combination, which defeats the point of a model chosen specifically
# for its readable coefficients. unrate_chg_12m is still computed in
# _fetch_raw_monthly_frame() below (harmless, unused) in case it's
# useful again later; it's just excluded from FEATURES.
#
# The remaining 4 features still have real correlation with each other
# (economic indicators always do -- it's the whole premise of this app's
# own correlation heatmap on the main dashboard). Rather than keep
# dropping features to chase perfectly independent inputs, coefficient
# instability from correlation is surfaced directly in the API response
# (see _correlation_notes) so it's traceable, not hidden, if a
# feature's contribution ever looks smaller or differently-signed than
# its own standalone relationship with recessions would suggest.
CORRELATION_NOTE_THRESHOLD = 0.35

LOOKAHEAD_MONTHS = 12

_MODEL_CACHE = {}
_CACHE_LOCK = threading.Lock()


def _series_to_monthly(raw, name):
    """Turn a raw [{date, value}] FRED response into a clean monthly Series.

    '.' (FRED's missing-value sentinel) becomes NaN via pd.to_numeric's
    errors="coerce", and resample("MS") snaps every series onto an
    identical month-start index so they align cleanly when concatenated.
    """
    df = pd.DataFrame(raw)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.set_index("date")["value"].sort_index().resample("MS").last()
    s.name = name
    return s


def _fetch_one(name, series_id):
    # Network I/O only. See the docstring below for why that split matters.
    return name, get_fred_data(series_id, limit=1000)


def _fetch_raw_monthly_frame():
    """Fetch + align all source series, and engineer the model's features.

    limit=1000 (~83 years) grabs each series' full history in one call --
    same "just ask for plenty" pattern the app's existing /api/sahm
    (limit=800) and /api/mortgage (limit=600) routes already use.

    The 7 series' HTTP fetches run in parallel via ThreadPoolExecutor --
    the same fix app.py's /api/all already applies for the same reason
    (these are blocking network calls, not CPU work, so the waits can
    overlap). This isn't just an optimization here: predict_current()
    calls this function fresh on every request (by design -- see its
    docstring), so a sequential 7-call fetch was slow enough on every
    single request, not just the first cold one, to occasionally exceed
    gunicorn's worker timeout and truncate the response mid-stream.

    IMPORTANT: only the raw HTTP fetch happens inside the worker
    threads. _series_to_monthly (pandas/numpy: DataFrame construction,
    to_datetime, resample) runs afterward, sequentially, on the main
    thread. An earlier version called _series_to_monthly from inside
    each thread, which crashed gunicorn workers in production with
    SIGSEGV -- numpy/pandas lean on BLAS libraries underneath that
    aren't guaranteed thread-safe for concurrent access on every
    platform, and Render's Linux environment hit that instability even
    though it never reproduced locally. Parallelizing the network calls
    gets nearly all of the same speedup (the network was the actual
    bottleneck) without touching numpy/pandas from more than one
    thread at a time.
    """
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(_fetch_one, FRED_SERIES.keys(), FRED_SERIES.values())
    raw_by_name = dict(results)
    frames = [_series_to_monthly(raw_by_name[name], name) for name in FRED_SERIES]
    df = pd.concat(frames, axis=1).sort_index()

    df["yield_spread"] = df["yield_long"] - df["yield_short"]
    df["unrate_chg_12m"] = df["unrate"] - df["unrate"].shift(12)
    df["cpi_yoy"] = df["cpi"].pct_change(12, fill_method=None) * 100

    return df


def fetch_training_frame():
    """Build the labeled training frame: features + forward-looking label.

    label[t] = 1 if a recession occurs in ANY of the next 12 months
    (t+1 .. t+12), else 0. Pandas rolling only looks backward, so we
    shift first, reverse, roll, then reverse back:

      1. shift(-1)          -> shifted[t] = USREC[t+1]
      2. iloc[::-1]          -> reverse chronological order
      3. rolling(12, min_periods=12).max()
                              -> at each row, the max over the current +
                                 next 11 rows of the *reversed* series,
                                 which (because of the shift) is
                                 max(USREC[t+1], ..., USREC[t+12])
      4. iloc[::-1]          -> reverse back to chronological order

    min_periods=12 (not 1) matters: without it, the most recent ~12
    months -- where we don't yet know what the next year holds -- would
    silently get labeled 0, which would be a fabricated answer rather
    than real ground truth. With min_periods=12 those rows become NaN
    and get dropped below by .dropna(), instead of being guessed at.
    """
    df = _fetch_raw_monthly_frame()
    df["label"] = (
        df["label_source"]
        .shift(-1)
        .iloc[::-1]
        .rolling(window=LOOKAHEAD_MONTHS, min_periods=LOOKAHEAD_MONTHS)
        .max()
        .iloc[::-1]
    )
    # .dropna() here does triple duty: drops the ~12 trailing
    # unknowable-label months, drops leading months missing the 12-month
    # lookback needed for unrate_chg_12m/cpi_yoy, and drops anything
    # before whichever source series has the shortest history -- so the
    # effective training window falls out of the data itself.
    return df[FEATURES + ["label"]].dropna()


def _count_recession_episodes(usrec):
    """Count distinct recession episodes (0->1 transitions) in USREC."""
    is_recession = usrec.fillna(0) >= 1
    starts = is_recession & ~is_recession.shift(1, fill_value=False)
    return int(starts.sum())


def _correlation_notes(corr):
    """For each feature, name its most-correlated *other* feature if that
    correlation is strong enough to plausibly distort its coefficient.

    This is what lets a confusing-looking contribution in the breakdown
    (e.g. a small or backwards-looking effect for a feature that's
    genuinely predictive on its own) be traced back to a real, visible
    cause instead of looking like a bug.
    """
    notes = {}
    for name in FEATURES:
        others = corr[name].drop(name)
        top_other = others.abs().idxmax()
        top_value = float(others[top_other])
        if abs(top_value) >= CORRELATION_NOTE_THRESHOLD:
            notes[name] = {
                "correlated_with": FEATURE_LABELS[top_other],
                "correlation": round(top_value, 3),
            }
    return notes


def train():
    df = fetch_training_frame()
    X = df[FEATURES].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression()
    model.fit(X_scaled, y)

    raw = _fetch_raw_monthly_frame()
    episodes = _count_recession_episodes(raw["label_source"].loc[df.index[0]:df.index[-1]])
    correlations = df[FEATURES].corr()

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_start": df.index.min().strftime("%Y-%m-%d"),
        "training_end": df.index.max().strftime("%Y-%m-%d"),
        "n_months": len(df),
        "n_recession_months": int(df["label"].sum()),
        "n_recession_episodes": episodes,
    }
    return {
        "model": model,
        "scaler": scaler,
        "metadata": metadata,
        "correlation_notes": _correlation_notes(correlations),
    }


def get_model():
    """Lazy, thread-safe, in-process model cache.

    Trained once on first use and cached for the process's lifetime --
    not on a timer and not on every request. See the plan doc for why:
    the underlying FRED series update at most monthly, and with only a
    handful of recession episodes in the training data, one more month
    of "normal" data barely moves the fit -- frequent retraining adds
    churn without making the model meaningfully smarter. A new fit only
    happens when the server process restarts.
    """
    with _CACHE_LOCK:
        if "bundle" not in _MODEL_CACHE:
            _MODEL_CACHE["bundle"] = train()
        return _MODEL_CACHE["bundle"]


def _feature_breakdown(model, x_scaled_row, raw_row, correlation_notes):
    """Turn fitted coefficients into a human-readable "what's driving this".

    After scaling, log-odds = intercept + sum(coefficient_i * z_i), so
    coefficient_i * z_i is literally each feature's own additive
    contribution to the log-odds, in the same units as every other
    feature -- exactly what a "what's driving this number" list needs.

    Each row also carries `correlation_note` when that feature shares
    strong correlation with another one in the model (see
    _correlation_notes) -- the traceable reason a contribution might
    look smaller, or point a different direction, than the feature's
    own standalone relationship with recession risk would suggest.
    """
    rows = []
    for i, name in enumerate(FEATURES):
        coef = float(model.coef_[0][i])
        z = float(x_scaled_row[i])
        contribution = coef * z
        rows.append({
            "feature": FEATURE_LABELS[name],
            "raw_value": round(float(raw_row[name]), 3),
            "coefficient": round(coef, 4),
            "contribution": round(contribution, 4),
            "direction": "increases" if contribution > 0 else "decreases",
            "correlation_note": correlation_notes.get(name),
        })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows


def predict_current():
    """The public entry point used by the /api/recession-model route.

    Deliberate asymmetry: the model (learned coefficients) is cached,
    but the feature values fed into it are fetched fresh from FRED on
    every call -- consistent with the rest of this app's "no caching,
    always live" behavior. Both metadata.trained_at and this
    prediction's own as_of date are returned so it's visible if they've
    drifted apart.
    """
    bundle = get_model()
    model, scaler, metadata = bundle["model"], bundle["scaler"], bundle["metadata"]

    latest = _fetch_raw_monthly_frame()[FEATURES].dropna().iloc[-1]
    x_scaled = scaler.transform(latest.values.reshape(1, -1))
    probability = float(model.predict_proba(x_scaled)[0][1])

    return {
        "probability": round(probability, 4),
        "as_of": latest.name.strftime("%Y-%m-%d"),
        "breakdown": _feature_breakdown(model, x_scaled[0], latest, bundle["correlation_notes"]),
        "metadata": metadata,
    }
