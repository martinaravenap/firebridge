"""Pick the clearest NASA GIBS dates for each site (cloud & no-data scoring on real tiles)."""
import urllib.request, math, io, json, sys, datetime as dt
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

BASE = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"

def xy(lat, lon, z):
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)
    return x, y

def fetch(layer, date, z, x, y, ext, tms):
    u = f"{BASE}/{layer}/default/{date}/{tms}/{z}/{y}/{x}.{ext}"
    try:
        return Image.open(io.BytesIO(urllib.request.urlopen(u, timeout=40).read())).convert("RGBA")
    except Exception:
        return None

def score(layer, date, lat, lon, z, ext, tms, falsecolor=False):
    x0, y0 = xy(lat, lon, z)
    ims = [fetch(layer, date, z, x, y, ext, tms) for x in (x0 - 1, x0) for y in (y0 - 1, y0)]
    if any(i is None for i in ims):
        return 0, 0, 0
    a = np.concatenate([np.asarray(i, dtype=float) for i in ims])
    rgb, alpha = a[..., :3], a[..., 3]
    valid = (alpha > 0) & (rgb.sum(-1) > 30)
    v = valid.mean()
    if v == 0:
        return 0, 0, 0
    px = rgb[valid]
    mx, mn = px.max(-1), px.min(-1)
    cloud = (((px[:, 1] > 135) & (px[:, 2] > 135)) if falsecolor else ((mn > 165) & (mx - mn < 45))).mean()
    return v * (1 - cloud) ** 2, v, cloud

def best(layer, dates, lat, lon, z, ext="jpg", tms="GoogleMapsCompatible_Level9", falsecolor=False):
    with ThreadPoolExecutor(12) as ex:
        res = list(ex.map(lambda d: (d, *score(layer, d, lat, lon, z, ext, tms, falsecolor)), dates))
    res.sort(key=lambda r: -r[1])
    return res

def drange(a, b, step=1):
    d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
    out = []
    while d0 <= d1:
        out.append(d0.isoformat()); d0 += dt.timedelta(days=step)
    return out
