"""Download + mosaic the SwissALTI3D terrain for the Klosters reach (Prättigau, GR).

Pulls the 2 m swisstopo SwissALTI3D tiles for a 4 x 3 km block around Klosters Platz/Dorf
(LV95 easting 2784-2787, northing 1193-1195), mosaics them, block-averages to a 10 m grid,
and writes data/klosters/dem10.npy + dem10.json for examples/klosters_flood_2005.py.

Public swisstopo STAC API; no authentication required. Requires rasterio.
"""

import json
import os
import re
import urllib.request

import numpy as np
import rasterio
from rasterio.merge import merge

HERE = os.path.dirname(__file__)
OUT = os.path.join(os.path.dirname(HERE), "data", "klosters")
STAC = "https://data.geo.admin.ch/api/stac/v0.9/collections/ch.swisstopo.swissalti3d/items"

TILES = [(e, n) for e in (2784, 2785, 2786, 2787) for n in (1193, 1194, 1195)]
BBOX = "9.84,46.855,9.92,46.895"
DOWNSAMPLE = 5  # 2 m -> 10 m


def main():
    os.makedirs(OUT, exist_ok=True)
    feats = json.load(urllib.request.urlopen(f"{STAC}?bbox={BBOX}&limit=200", timeout=90))["features"]
    hrefs = {}
    for f in feats:
        m = re.search(r"_(\d{4})-(\d{4})", f["id"])
        key = (int(m.group(1)), int(m.group(2))) if m else None
        if key in TILES:
            for k, a in f["assets"].items():
                if k.endswith("_2_2056_5728.tif"):
                    hrefs[key] = a["href"]
    missing = set(TILES) - set(hrefs)
    if missing:
        raise SystemExit(f"STAC did not return tiles: {sorted(missing)}")

    paths = []
    for key, href in sorted(hrefs.items()):
        p = os.path.join(OUT, f"{key[0]}-{key[1]}.tif")
        if not (os.path.exists(p) and os.path.getsize(p) > 0):
            print("downloading", os.path.basename(p))
            urllib.request.urlretrieve(href, p)
        paths.append(p)

    mosaic, tr = merge([rasterio.open(p) for p in paths])
    dem = mosaic[0].astype(float)
    f = DOWNSAMPLE
    ny = (dem.shape[0] // f) * f
    nx = (dem.shape[1] // f) * f
    dem10 = dem[:ny, :nx].reshape(ny // f, f, nx // f, f).mean(axis=(1, 3))

    np.save(os.path.join(OUT, "dem10.npy"), dem10)
    meta = dict(E0=float(tr.c), N0=float(tr.f), res=float(tr.a * f),
                nrows=int(dem10.shape[0]), ncols=int(dem10.shape[1]))
    json.dump(meta, open(os.path.join(OUT, "dem10.json"), "w"))
    print(f"wrote dem10.npy {dem10.shape} @ {meta['res']:.0f} m, "
          f"elev {dem10.min():.0f}-{dem10.max():.0f} m")


if __name__ == "__main__":
    main()
