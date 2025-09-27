import os
import sys
import json
from datetime import datetime, timedelta

from utils import build_params, fetch_data

OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "rtr_data.json")
STEP_DAYS = 3  # Nombre de jours par tranche
START_DATE = datetime.today() - timedelta(days=200)
END_DATE = datetime.today()


def save_to_json(data, filename):
    """Save the data dict into a JSON file with indentation."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fetch_all_data(start_date, end_date, step_days=30):
    """Fetch data from API by slicing the date range into steps."""
    all_results = []
    current = start_date

    while current < end_date:
        next_date = min(current + timedelta(days=step_days), end_date)
        print(f"Fetching from {current.date()} to {next_date.date()}...")

        try:
            params = build_params(current, next_date)
            data = fetch_data(params)
            results = data.get("results", [])
            print(f"  ➜ Retrieved {len(results)} results.")
            all_results.extend(results)

        except Exception as e:
            print(f" Error fetching from {current} to {next_date}: {e}", file=sys.stderr)

        current = next_date

    return {"results": all_results}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Fetching RTR data from {START_DATE.date()} to {END_DATE.date()} in {STEP_DAYS}-day steps...")
    all_data = fetch_all_data(START_DATE, END_DATE, STEP_DAYS)

    print(f"\nTotal results collected: {len(all_data['results'])}")
    save_to_json(all_data, OUTPUT_FILE)
    print(f"✅ Data saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
