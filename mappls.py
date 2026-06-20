
import os
import time
import requests

# ---- endpoints (verify against your console docs if selftest flags them) ----
TOKEN_URL = "https://outpost.mappls.com/api/security/oauth/token"
ROUTE_URL = "https://apis.mappls.com/advancedmaps/v1/{key}/route_adv/driving/{coords}"
GEOCODE_URL = "https://atlas.mappls.com/api/places/geocode"
BASE_TILE_URL = "https://apis.mappls.com/advancedmaps/v1/{key}/tile/{z}/{x}/{y}.png"
TRAFFIC_TILE_URL = "https://apis.mappls.com/advancedmaps/v1/{key}/traffic/{z}/{x}/{y}.png"

_TOKEN_CACHE = {"token": None, "expires_at": 0}


def _decode_polyline(s, precision=5):
    """Decode a Google/OSRM encoded polyline into [(lat, lon), ...]. No external dep."""
    if not s:
        return []
    coords, index, lat, lng = [], 0, 0, 0
    factor = float(10 ** precision)
    while index < len(s):
        for is_lng in (False, True):
            shift, result = 0, 0
            while True:
                b = ord(s[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if is_lng:
                lng += d
            else:
                lat += d
        coords.append((lat / factor, lng / factor))
    return coords


class MapplsClient:
    def __init__(self):
        self.client_id = os.environ.get("MAPPLS_CLIENT_ID", "").strip()
        self.client_secret = os.environ.get("MAPPLS_CLIENT_SECRET", "").strip()
        self.rest_key = os.environ.get("MAPPLS_REST_KEY", "").strip()

    def is_configured(self):
        return bool(self.rest_key) and bool(self.client_id) and bool(self.client_secret)

    # ------------------------------------------------------------------ #
    def _token(self):
        if _TOKEN_CACHE["token"] and time.time() < _TOKEN_CACHE["expires_at"]:
            return _TOKEN_CACHE["token"]
        if not (self.client_id and self.client_secret):
            return None
        try:
            r = requests.post(TOKEN_URL, data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }, timeout=10)
            r.raise_for_status()
            j = r.json()
            _TOKEN_CACHE["token"] = j["access_token"]
            _TOKEN_CACHE["expires_at"] = time.time() + int(j.get("expires_in", 3600)) - 60
            return _TOKEN_CACHE["token"]
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def route(self, o_lat, o_lon, d_lat, d_lon, alternatives=False):
        """Real road route(s). Returns list of {distance_m, duration_s, coords} or None."""
        if not self.rest_key:
            return None
        coords = f"{o_lon},{o_lat};{d_lon},{d_lat}"
        url = ROUTE_URL.format(key=self.rest_key, coords=coords)
        try:
            r = requests.get(url, params={
                "geometries": "polyline", "overview": "full",
                "alternatives": "true" if alternatives else "false", "steps": "false",
            }, timeout=12)
            r.raise_for_status()
            j = r.json()
            out = []
            for rt in j.get("routes", []):
                out.append({
                    "distance_m": rt.get("distance"),
                    "duration_s": rt.get("duration"),
                    "coords": _decode_polyline(rt.get("geometry", "")),
                })
            return out or None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def live_congestion_factor(self, o_lat, o_lon, d_lat, d_lon):
        """Live congestion as a ratio >= 1.0 (traffic time / free-flow time) along the
        straight O->D road route. Returns float or None.

        Mappls route duration reflects current conditions; we compare it to a free-flow
        estimate (distance / typical urban free-flow speed) to get a live multiplier.
        """
        routes = self.route(o_lat, o_lon, d_lat, d_lon, alternatives=False)
        if not routes:
            return None
        rt = routes[0]
        dist_m, dur_s = rt.get("distance_m"), rt.get("duration_s")
        if not dist_m or not dur_s:
            return None
        free_flow_speed_mps = 8.33          # ~30 km/h urban free-flow baseline
        free_flow_s = dist_m / free_flow_speed_mps
        factor = dur_s / max(free_flow_s, 1.0)
        return max(1.0, round(factor, 2))

    # ------------------------------------------------------------------ #
    def geocode(self, address):
        """Address -> (lat, lon) or None."""
        tok = self._token()
        if not tok:
            return None
        try:
            r = requests.get(GEOCODE_URL, headers={"Authorization": f"Bearer {tok}"},
                             params={"address": address, "itemCount": 1}, timeout=10)
            r.raise_for_status()
            j = r.json()
            res = (j.get("copResults") or j.get("results") or [None])[0]
            if not res:
                return None
            return float(res["latitude"]), float(res["longitude"])
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def base_tile_url(self):
        return BASE_TILE_URL.format(key=self.rest_key, z="{z}", x="{x}", y="{y}") if self.rest_key else None

    def traffic_tile_url(self):
        return TRAFFIC_TILE_URL.format(key=self.rest_key, z="{z}", x="{x}", y="{y}") if self.rest_key else None


_CLIENT = None
def client():
    """Singleton accessor."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = MapplsClient()
    return _CLIENT


def _selftest():
    c = client()
    print("\n=== Mappls self-test ===")
    print(f"client_id set:     {bool(c.client_id)}")
    print(f"client_secret set: {bool(c.client_secret)}")
    print(f"rest_key set:      {bool(c.rest_key)}")
    if not c.is_configured():
        print("\n[FAIL] credentials missing — set MAPPLS_CLIENT_ID / _SECRET / _REST_KEY. "
              "System will run in fallback (historical/OSM) mode.\n")
        return
    print("\n[1] OAuth token ...", end=" ")
    tok = c._token()
    print("OK" if tok else "FAIL (check client_id/secret or TOKEN_URL)")

    # central Bengaluru sample O->D
    o = (12.9716, 77.5946); d = (12.9784, 77.6408)
    print("[2] routing      ...", end=" ")
    rts = c.route(*o, *d)
    if rts:
        print(f"OK ({rts[0]['distance_m']} m, {len(rts[0]['coords'])} pts)")
    else:
        print("FAIL (check MAPPLS_REST_KEY or ROUTE_URL)")

    print("[3] live traffic ...", end=" ")
    f = c.live_congestion_factor(*o, *d)
    print(f"OK (congestion factor {f}x)" if f else "FAIL")

    print("[4] geocode      ...", end=" ")
    g = c.geocode("MG Road, Bengaluru")
    print(f"OK {g}" if g else "FAIL (check GEOCODE_URL/token)")

    print("\nIf any step says FAIL, update its URL constant at the top of mappls.py "
          "from https://apis.mappls.com docs. Other features still work independently.\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        print("usage: python mappls.py --selftest")