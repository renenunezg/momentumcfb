import os
import time

import requests

from backend.config import CFBD_BASE_URL


class CFBDError(RuntimeError):
    pass


class CFBDClient:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("CFBD_API_KEY")
        if not key:
            raise CFBDError("CFBD_API_KEY is not set")
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {key}"

    def get(self, path: str, params: dict | None = None, retries: int = 3) -> list:
        url = f"{CFBD_BASE_URL}{path}"
        for attempt in range(retries):
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise CFBDError(f"GET {path} rate limited after {retries} attempts")
            if resp.status_code != 200:
                raise CFBDError(f"GET {path} returned {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            if not isinstance(data, list):
                raise CFBDError(f"GET {path} returned a non-list payload")
            return data
