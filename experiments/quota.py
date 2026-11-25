"""Print the share of today's OpenAlex funds spent, daily budget plus prepaid credit (0.9592 = 96% spent).
Exit 1 if it cannot be read."""
import json
import os
import sys
import httpx2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from researchreadernerd.cli import load_dotenv  # noqa: E402

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
try:
    r = httpx2.get("https://api.openalex.org/rate-limit", params={"api_key": os.environ["OPENALEX_API_KEY"]}, timeout=20)
    d = r.json()["rate_limit"]
    # Funds left today: the daily budget's remainder plus prepaid credit (OpenAlex reports the prepaid balance as
    # what is left, so there is no fixed total). Printed as a fraction "spent" of one dollar: 0.95 = 5 cents left,
    # 0.0 = a dollar or more left. The runner stops a step at 0.95 and starts when below 0.5 (50 cents or more).
    left = d["daily_remaining_usd"] + (d.get("prepaid_remaining_usd") or 0)
    print(f"{1 - min(left, 1.0):.4f}")
except Exception as e:  # noqa: BLE001
    print(f"error {e}", file=sys.stderr)
    sys.exit(1)
