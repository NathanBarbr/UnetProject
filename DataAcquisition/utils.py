# utils.py

import requests
from config import API_URL, DEFAULT_MAX_RESULTS, DEFAULT_TIME_RANGE_DAYS
from datetime import datetime, timedelta

def get_default_dates():
    """Return the default [start, end] UTC timestamps (last DEFAULT_TIME_RANGE_DAYS days)."""
    end = datetime.utcnow()
    start = end - timedelta(days=DEFAULT_TIME_RANGE_DAYS)
    return start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S")

def build_params(start_date, end_date, max_results=DEFAULT_MAX_RESULTS):
    """Build and return the query parameters dict for the RTR API."""
    return {
        "time": f">{start_date}&<{end_date}",
        "format": "json",
        "max_results": max_results,
        # Add fields if you need extra metrics:
        # "fields": "download_kbit,upload_kbit,ping_ms,signal_strength,lte_rsrp,lte_rsrq,lat,long,loc_accuracy,platform,provider_name,model"
    }

def fetch_data(params):
    """
    Perform the GET request to the API and return parsed JSON.
    Raises on network errors or invalid JSON.
    """
    resp = requests.get(API_URL, params=params)
    resp.raise_for_status()

    try:
        data = resp.json()
    except ValueError:
        raise RuntimeError(f"Unable to parse JSON from response: {resp.text[:200]}")

    return data
